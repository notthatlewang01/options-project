"""Unit tests for `spxrnd.store.writer`, `.schema` and `.catalog`."""

from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime

import pyarrow.parquet as pq
import pytest

from spxrnd.ingest.payload import parse as parse_payload
from spxrnd.store import archive, catalog, schema, writer

from ..conftest import FIXTURES, load_header

CLOSE_STEM = "_SPX_2026-08-11T20-37-03Z"


@pytest.fixture
def raw_dir(tmp_path, trimmed_chain):
    """Three captures: the trimmed close chain twice, plus a headerless one."""
    d = tmp_path / "raw"
    d.mkdir()
    (d / f"{CLOSE_STEM}.json").write_text(json.dumps(trimmed_chain))
    (d / "_SPX_2026-08-11T16-11-33Z.json").write_text(json.dumps(trimmed_chain))
    (d / "_SPX_2026-08-13T01-25-33Z.json.gz").write_bytes(
        gzip.compress(json.dumps(trimmed_chain).encode(), mtime=0)
    )
    return d


@pytest.fixture
def curated_dir(tmp_path):
    return tmp_path / "curated"


@pytest.fixture
def built(raw_dir, curated_dir):
    catalog.backfill(raw_dir, curated_dir)
    return curated_dir


class TestSchema:
    def test_declared_not_inferred(self):
        """Inference types a column from whatever the first file contains, and
        two captures that inferred differently will not union."""
        assert schema.SCHEMA.field("bid").type == schema.pa.float64()
        assert schema.SCHEMA.field("bid_size").type == schema.pa.float64()

    def test_capture_columns_are_non_nullable(self):
        for name in schema.CAPTURE_COLUMNS:
            assert not schema.SCHEMA.field(name).nullable, name

    def test_contract_columns_are_non_nullable(self):
        for name in schema.CONTRACT_COLUMNS:
            assert not schema.SCHEMA.field(name).nullable, name

    def test_no_column_is_named_after_a_sql_keyword(self):
        """`right` would need quoting in every query that touched it."""
        reserved = {"right", "left", "order", "group", "select", "table", "all"}
        assert not (set(schema.SCHEMA.names) & reserved)

    def test_expiry_is_a_date_not_a_string(self):
        """A string expiry is how tenor arithmetic quietly gets skipped."""
        assert schema.SCHEMA.field("expiry").type == schema.pa.date32()

    def test_timestamps_are_millisecond_utc(self):
        """Parquet's floor is ms; declaring "s" would disagree with the file."""
        for name in ("capture_utc", "index_last_trade"):
            field = schema.SCHEMA.field(name)
            assert field.type.unit == "ms"
            assert field.type.tz == "UTC"

    def test_null_and_zero_stay_distinct(self):
        """ "not reported" and "reported as zero" are different claims, and the
        iv == 0.0 sentinel is exactly where collapsing them would hurt."""
        assert schema._number(None) is None
        assert schema._number(0.0) == 0.0
        assert schema._number(0) == 0.0

    def test_booleans_are_not_numbers(self):
        assert schema._number(True) is None

    def test_unparseable_values_become_null(self):
        assert schema._number("7728.2") is None
        assert schema._text("") is None
        assert schema._text(42) is None


class TestBuildTable:
    def test_row_count_matches_the_chain(self, trimmed_chain):
        snap = parse_payload(trimmed_chain, ticker="_SPX")
        table, unparsed = writer.build_table(snap, captured_at=datetime.now(UTC))
        assert table.num_rows == 674
        assert unparsed == 0

    def test_conforms_to_the_declared_schema(self, trimmed_chain):
        snap = parse_payload(trimmed_chain, ticker="_SPX")
        table, _ = writer.build_table(snap, captured_at=datetime.now(UTC))
        assert table.schema.equals(schema.SCHEMA)

    def test_capture_columns_repeat_down_every_row(self, trimmed_chain):
        snap = parse_payload(trimmed_chain, ticker="_SPX")
        table, _ = writer.build_table(snap, captured_at=datetime.now(UTC))
        assert len(set(table.column("underlying_price").to_pylist())) == 1
        assert table.column("underlying_price")[0].as_py() == 7728.2002

    def test_unparseable_symbols_are_counted_not_hidden(self, trimmed_chain):
        """Dropping contracts silently is how a chain loses an expiry."""
        corrupted = json.loads(json.dumps(trimmed_chain))
        corrupted["data"]["options"][0]["option"] = "NOT-AN-OSI-SYMBOL"
        snap = parse_payload(corrupted, ticker="_SPX")
        table, unparsed = writer.build_table(snap, captured_at=datetime.now(UTC))
        assert unparsed == 1
        assert table.num_rows == 673

    def test_the_iv_sentinel_survives_as_zero_not_null(self, trimmed_chain):
        """The store records what was served. Judging it is `curate`'s job."""
        snap = parse_payload(trimmed_chain, ticker="_SPX")
        table, _ = writer.build_table(snap, captured_at=datetime.now(UTC))
        ivs = table.column("iv").to_pylist()
        assert 0.0 in ivs
        assert None not in ivs


class TestWriteCapture:
    def test_writes_a_parquet_named_for_the_capture(self, raw_dir, curated_dir):
        capture = archive.scan(raw_dir)[0]
        result = writer.write_capture(capture, curated_dir)
        assert result.path.name == f"{capture.stem}.parquet"
        assert result.path.exists()

    def test_output_path_is_deterministic(self, raw_dir, curated_dir):
        capture = archive.scan(raw_dir)[0]
        a = writer.write_capture(capture, curated_dir).path
        b = writer.write_capture(capture, curated_dir).path
        assert a == b

    def test_rewriting_overwrites_rather_than_duplicating(self, raw_dir, curated_dir):
        capture = archive.scan(raw_dir)[0]
        writer.write_capture(capture, curated_dir)
        writer.write_capture(capture, curated_dir)
        assert len(list(curated_dir.glob("*.parquet"))) == 1

    def test_overwrite_false_skips_existing(self, raw_dir, curated_dir):
        capture = archive.scan(raw_dir)[0]
        writer.write_capture(capture, curated_dir)
        assert writer.write_capture(capture, curated_dir, overwrite=False) is None

    def test_reads_compressed_and_uncompressed_alike(self, raw_dir, curated_dir):
        for capture in archive.scan(raw_dir):
            assert writer.write_capture(capture, curated_dir).rows == 674

    def test_capture_time_comes_from_the_filename(self, raw_dir, curated_dir):
        """The payload does not know when we asked for it."""
        capture = [c for c in archive.scan(raw_dir) if c.stem == CLOSE_STEM][0]
        result = writer.write_capture(capture, curated_dir)
        stamps = set(
            pq.read_table(result.path, columns=["capture_utc"])
            .column("capture_utc")
            .to_pylist()
        )
        assert stamps == {datetime(2026, 8, 11, 20, 37, 3, tzinfo=UTC)}

    def test_no_temp_file_survives_a_failed_write(
        self, raw_dir, curated_dir, monkeypatch
    ):
        """A reader must never find a half-written Parquet."""
        monkeypatch.setattr(
            writer.pq,
            "write_table",
            lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")),
        )
        with pytest.raises(OSError):
            writer.write_capture(archive.scan(raw_dir)[0], curated_dir)
        assert list(curated_dir.glob("*")) == []


class TestBackfill:
    def test_converts_every_capture(self, raw_dir, curated_dir):
        report = catalog.backfill(raw_dir, curated_dir)
        assert len(report.written) == 3
        assert report.failed == []

    def test_is_idempotent(self, raw_dir, curated_dir):
        catalog.backfill(raw_dir, curated_dir)
        first = sorted(p.name for p in curated_dir.glob("*.parquet"))
        catalog.backfill(raw_dir, curated_dir)
        assert sorted(p.name for p in curated_dir.glob("*.parquet")) == first

    def test_resumes_without_redoing_work(self, raw_dir, curated_dir):
        catalog.backfill(raw_dir, curated_dir)
        second = catalog.backfill(raw_dir, curated_dir, overwrite=False)
        assert len(second.skipped) == 3
        assert second.written == []

    def test_a_bad_capture_does_not_abort_the_run(self, raw_dir, curated_dir):
        """The other captures are still worth converting."""
        (raw_dir / "_SPX_2026-08-14T20-37-03Z.json").write_text('{"garbage": true}')
        report = catalog.backfill(raw_dir, curated_dir)
        assert len(report.written) == 3
        assert len(report.failed) == 1
        assert "PayloadError" in report.failed[0][1]

    def test_totals_are_reported(self, raw_dir, curated_dir):
        report = catalog.backfill(raw_dir, curated_dir)
        assert report.total_rows == 3 * 674
        assert report.total_unparsed == 0
        assert report.total_bytes > 0


class TestDivergence:
    def test_aligned_after_a_full_backfill(self, raw_dir, curated_dir):
        catalog.backfill(raw_dir, curated_dir)
        assert catalog.divergence(raw_dir, curated_dir).aligned

    def test_detects_a_capture_that_never_reached_the_curated_layer(
        self, raw_dir, curated_dir
    ):
        """Seed regression 6, as a standing check rather than an accident."""
        catalog.backfill(raw_dir, curated_dir)
        (curated_dir / f"{CLOSE_STEM}.parquet").unlink()
        div = catalog.divergence(raw_dir, curated_dir)
        assert div.missing_curated == [CLOSE_STEM]
        assert not div.aligned

    def test_detects_an_orphaned_parquet(self, raw_dir, curated_dir):
        """The raw archive is the source of truth and should only grow."""
        catalog.backfill(raw_dir, curated_dir)
        [c for c in archive.scan(raw_dir) if c.stem == CLOSE_STEM][0].path.unlink()
        div = catalog.divergence(raw_dir, curated_dir)
        assert div.orphan_curated == [CLOSE_STEM]

    def test_an_empty_curated_layer_reports_everything_missing(
        self, raw_dir, curated_dir
    ):
        curated_dir.mkdir()
        assert len(catalog.divergence(raw_dir, curated_dir).missing_curated) == 3


class TestCatalogQueries:
    def test_quotes_view_exposes_every_row(self, built):
        con = catalog.connect(built)
        assert con.execute("SELECT count(*) FROM quotes").fetchone()[0] == 3 * 674

    def test_captures_view_is_one_row_per_capture(self, built):
        con = catalog.connect(built)
        assert con.execute("SELECT count(*) FROM captures").fetchone()[0] == 3

    def test_timestamps_render_in_utc_regardless_of_host(self, built):
        """Otherwise the same query prints different times in Chicago and
        Frankfurt, and a reader cannot tell which they are seeing."""
        con = catalog.connect(built)
        got = con.execute(
            "SELECT capture_utc FROM captures ORDER BY capture_utc"
        ).fetchall()[1][0]
        assert got == datetime(2026, 8, 11, 20, 37, 3, tzinfo=UTC)

    def test_index_print_round_trips_to_eastern(self, built):
        """16:14:59 ET is 20:14:59 UTC. Getting this backwards would shift every
        tenor downstream by four hours."""
        con = catalog.connect(built)
        row = con.execute("""
            SELECT index_last_trade,
                   index_last_trade AT TIME ZONE 'America/New_York'
            FROM captures LIMIT 1
        """).fetchone()
        assert row[0] == datetime(2026, 8, 11, 20, 14, 59, tzinfo=UTC)
        assert str(row[1]) == "2026-08-11 16:14:59"

    def test_root_is_queryable_without_quoting(self, built):
        con = catalog.connect(built)
        roots = con.execute("SELECT DISTINCT root FROM quotes ORDER BY root").fetchall()
        assert [r[0] for r in roots] == ["SPX", "SPXW"]

    def test_option_right_needs_no_quoting(self, built):
        con = catalog.connect(built)
        rights = con.execute(
            "SELECT DISTINCT option_right FROM quotes ORDER BY option_right"
        ).fetchall()
        assert [r[0] for r in rights] == ["call", "put"]

    def test_the_root_collision_is_visible_in_sql(self, built):
        con = catalog.connect(built)
        n = con.execute("""
            SELECT count(*) FROM (
              SELECT expiry, option_right, strike FROM quotes
              WHERE expiry = '2026-08-21'
              GROUP BY 1,2,3 HAVING count(DISTINCT root) > 1)
        """).fetchone()[0]
        assert n > 0

    def test_helpful_error_when_backfill_has_not_run(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="make backfill"):
            catalog.connect(tmp_path / "empty")


class TestFullChainIntegration:
    """One test against the real 30,692-option capture, so scale behaviour is
    exercised rather than assumed."""

    @pytest.fixture
    def full_raw(self, tmp_path):
        d = tmp_path / "raw"
        d.mkdir()
        (d / f"{CLOSE_STEM}.json.gz").write_bytes(
            (FIXTURES / "chain_full_close.json.gz").read_bytes()
        )
        return d

    def test_reconciles_against_the_measured_chain(self, full_raw, tmp_path):
        catalog.backfill(full_raw, tmp_path / "curated")
        con = catalog.connect(tmp_path / "curated")
        row = con.execute("""
            SELECT count(*), count(DISTINCT expiry),
                   sum(root='SPX'), sum(root='SPXW'), any_value(underlying_price)
            FROM quotes
        """).fetchone()
        assert row == (30692, 55, 10156, 20536, 7728.2002)

    def test_every_contract_survives_the_round_trip(self, full_raw, tmp_path):
        report = catalog.backfill(full_raw, tmp_path / "curated")
        assert report.total_rows == 30_692
        assert report.total_unparsed == 0

    def test_parquet_is_much_smaller_than_the_raw_json(self, full_raw, tmp_path):
        report = catalog.backfill(full_raw, tmp_path / "curated")
        raw_bytes = len(gzip.decompress(next(full_raw.iterdir()).read_bytes()))
        assert report.total_bytes < raw_bytes / 4

    def test_the_iv_sentinel_count_survives_to_sql(self, full_raw, tmp_path):
        catalog.backfill(full_raw, tmp_path / "curated")
        con = catalog.connect(tmp_path / "curated")
        n = con.execute("SELECT count(*) FROM quotes WHERE iv = 0.0").fetchone()[0]
        assert n == 2160

    def test_feed_timestamp_is_populated_in_every_row(self, full_raw, tmp_path):
        """Seed regression 1, asserted over a full real capture. The retired
        CSVs had this blank in all 890,000 of their rows."""
        catalog.backfill(full_raw, tmp_path / "curated")
        con = catalog.connect(tmp_path / "curated")
        blank = con.execute(
            "SELECT count(*) FROM quotes WHERE feed_timestamp IS NULL "
            "OR feed_timestamp = ''"
        ).fetchone()[0]
        assert blank == 0


def test_headers_with_no_options_produce_an_empty_but_valid_parquet(tmp_path):
    """A payload can legitimately carry no chain; that is a data question for
    `curate`, not a write failure here."""
    d = tmp_path / "raw"
    d.mkdir()
    (d / "_SPX_2026-08-11T16-11-33Z.json").write_text(
        json.dumps(load_header("2026-08-11T16-11-33Z"))
    )
    report = catalog.backfill(d, tmp_path / "curated")
    assert len(report.written) == 1
    assert report.total_rows == 0
    table = pq.read_table(report.written[0].path)
    assert table.schema.equals(schema.SCHEMA)
