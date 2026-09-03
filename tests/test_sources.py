# tests/test_sources.py

"""
Tests for autofloods.sources -- all network calls mocked, no live
credentials required.

Priority is authenticate(): this is the exact gap that let a real bug
(pystac_client.Client.open() called with an unsupported timeout=
kwarg against the pinned pystac-client==0.6.1) ship undetected in both
MPCSource and OPERASource for multiple releases. These tests assert
the exact call signature made to Client.open(), so a future
reintroduction of an unsupported kwarg fails here instead of only at
runtime against a real pinned dependency.
"""

from unittest.mock import MagicMock, patch

import pytest

from autofloods.sources.mpc import MPCSource
from autofloods.sources.opera import OPERASource, OperaPass


class TestMPCSourceAuthenticate:
    def test_authenticate_calls_client_open_with_only_supported_kwargs(self):
        """
        Regression test for the shipped bug: Client.open() was called
        with timeout=(15, 30), which pinned pystac-client==0.6.1's
        Client.open() does not accept. Bind the mock's spec to the
        real Client.open signature so an unsupported kwarg raises
        TypeError here, exactly like it did against the real pinned
        dependency.
        """
        with patch("autofloods.sources.mpc.pystac_client.Client.open",
                   autospec=True) as mock_open:
            src = MPCSource()
            src.authenticate()

        assert mock_open.called
        _, kwargs = mock_open.call_args
        assert "timeout" not in kwargs

    def test_authenticate_sets_subscription_key_when_provided(self):
        with patch("autofloods.sources.mpc.pystac_client.Client.open") as mock_open, \
             patch("planetary_computer.settings.set_subscription_key") as mock_set_key:
            src = MPCSource(subscription_key="fake-key-123")
            src.authenticate()

        mock_set_key.assert_called_once_with("fake-key-123")
        assert mock_open.called

    def test_authenticate_proceeds_anonymously_without_key(self):
        with patch("autofloods.sources.mpc.pystac_client.Client.open") as mock_open, \
             patch("planetary_computer.settings.set_subscription_key") as mock_set_key, \
             patch.dict("os.environ", {}, clear=True):
            src = MPCSource(subscription_key=None)
            src.authenticate()

        mock_set_key.assert_not_called()
        assert mock_open.called

    def test_subscription_key_falls_back_to_env_var(self):
        with patch.dict("os.environ", {"MPC_SUBSCRIPTION_KEY": "env-key"}, clear=True):
            src = MPCSource()
        assert src._subscription_key == "env-key"

    def test_explicit_key_overrides_env_var(self):
        with patch.dict("os.environ", {"MPC_SUBSCRIPTION_KEY": "env-key"}, clear=True):
            src = MPCSource(subscription_key="explicit-key")
        assert src._subscription_key == "explicit-key"

    def test_authenticate_sets_catalog(self):
        fake_catalog = MagicMock()
        with patch("autofloods.sources.mpc.pystac_client.Client.open", return_value=fake_catalog):
            src = MPCSource()
            src.authenticate()
        assert src._catalog is fake_catalog


class TestMPCSourceSign:
    def test_sign_strips_existing_signature_before_resigning(self):
        already_signed = "https://example.blob.core.windows.net/scene.tif?st=1&se=2&sp=r"
        with patch("planetary_computer.sign") as mock_sign:
            mock_sign.return_value = "https://example.blob.core.windows.net/scene.tif?st=NEW"
            src = MPCSource()
            src.sign(already_signed)

        mock_sign.assert_called_once_with("https://example.blob.core.windows.net/scene.tif")

    def test_vv_vh_hrefs_reads_configured_asset_keys(self):
        item = MagicMock()
        item.assets = {
            "vv": MagicMock(href="https://example/vv.tif"),
            "vh": MagicMock(href="https://example/vh.tif"),
        }
        src = MPCSource()
        vv_href, vh_href = src.vv_vh_hrefs(item)
        assert vv_href == "https://example/vv.tif"
        assert vh_href == "https://example/vh.tif"


class TestOPERASourceAuthenticate:
    def test_authenticate_calls_client_open_with_only_supported_kwargs(self):
        """
        Regression test for the shipped bug -- see module docstring and
        TestMPCSourceAuthenticate's equivalent test. autospec binds the
        mock to the real (pinned) Client.open signature, so passing an
        unsupported kwarg like timeout= raises TypeError here.
        """
        with patch("autofloods.sources.opera.pystac_client.Client.open",
                   autospec=True) as mock_open:
            src = OPERASource()
            src.authenticate()

        assert mock_open.called
        args, kwargs = mock_open.call_args
        assert "timeout" not in kwargs
        # CMR_ASF_STAC_URL is the sole positional/keyword URL argument
        from autofloods.sources.opera import CMR_ASF_STAC_URL
        assert CMR_ASF_STAC_URL in args or kwargs.get("url") == CMR_ASF_STAC_URL

    def test_authenticate_sets_catalog_and_session(self):
        fake_catalog = MagicMock()
        with patch("autofloods.sources.opera.pystac_client.Client.open", return_value=fake_catalog):
            src = OPERASource()
            src.authenticate()
        assert src._catalog is fake_catalog
        assert src._session is not None

    def test_authenticate_warns_when_no_netrc(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setenv("HOME", str(tmp_path))  # no .netrc in this fake home
        with patch("autofloods.sources.opera.pystac_client.Client.open"):
            src = OPERASource()
            with caplog.at_level("WARNING"):
                src.authenticate()
        assert any("netrc" in record.message for record in caplog.records)


class TestOPERASourceContract:
    def test_vv_vh_hrefs_always_raises(self):
        src = OPERASource()
        with pytest.raises(NotImplementedError):
            src.vv_vh_hrefs(MagicMock())

    def test_dem_search_delegates_to_internal_mpc_source(self):
        src = OPERASource()
        with patch.object(src._dem_source, "search_dem", return_value=["fake_item"]) as mock_search:
            result = src.search_dem({"type": "Polygon", "coordinates": []})
        mock_search.assert_called_once()
        assert result == ["fake_item"]


def _fake_item(item_id, has_vv=True, has_vh=True):
    from shapely.geometry import box, mapping

    item = MagicMock()
    item.id = item_id
    item.geometry = mapping(box(0, 0, 1, 1))
    assets = {}
    if has_vv:
        assets["0_VV"] = MagicMock()
    if has_vh:
        assets["0_VH"] = MagicMock()
    item.assets = assets
    return item


class TestOPERASourceSearchSentinel1:
    """
    search_sentinel1's reprocessing-dedup and same-date pass-grouping
    logic, exercised directly against a mocked catalog (no real STAC
    search) with synthetic items shaped like real OPERA RTC-S1 item
    IDs: OPERA_L2_RTC-S1_T<tile>-<burst>-<swath>_<acquisition>_<processing>_S1A_30_v1.0
    """

    def test_dedups_reprocessed_granules_keeping_latest(self):
        older = _fake_item("OPERA_L2_RTC-S1_T001-BURST-IW1_20240115T000000Z_20240116T000000Z_S1A_30_v1.0")
        newer = _fake_item("OPERA_L2_RTC-S1_T001-BURST-IW1_20240115T000000Z_20240118T000000Z_S1A_30_v1.0")

        fake_catalog = MagicMock()
        fake_catalog.search.return_value.items.return_value = [older, newer]

        src = OPERASource()
        src._catalog = fake_catalog

        import datetime
        passes = src.search_sentinel1(
            bbox={"type": "Polygon", "coordinates": []},
            start_date=datetime.date(2024, 1, 1),
            end_date=datetime.date(2024, 1, 31),
        )

        assert len(passes) == 1
        assert len(passes[0].bursts) == 1
        assert passes[0].bursts[0].id == newer.id  # kept the later processing_time

    def test_groups_same_date_bursts_into_one_pass(self):
        burst_a = _fake_item("OPERA_L2_RTC-S1_T001-A-IW1_20240115T000000Z_20240116T000000Z_S1A_30_v1.0")
        burst_b = _fake_item("OPERA_L2_RTC-S1_T001-B-IW2_20240115T000000Z_20240116T000000Z_S1A_30_v1.0")
        different_date = _fake_item("OPERA_L2_RTC-S1_T001-C-IW1_20240120T000000Z_20240121T000000Z_S1A_30_v1.0")

        fake_catalog = MagicMock()
        fake_catalog.search.return_value.items.return_value = [burst_a, burst_b, different_date]

        src = OPERASource()
        src._catalog = fake_catalog

        import datetime
        passes = src.search_sentinel1(
            bbox={"type": "Polygon", "coordinates": []},
            start_date=datetime.date(2024, 1, 1),
            end_date=datetime.date(2024, 1, 31),
        )

        assert len(passes) == 2
        by_id = {p.id: p for p in passes}
        assert len(by_id["OPERA_PASS_20240115"].bursts) == 2
        assert len(by_id["OPERA_PASS_20240120"].bursts) == 1

    def test_filters_items_missing_either_band(self):
        missing_vh = _fake_item("OPERA_L2_RTC-S1_T001-A-IW1_20240115T000000Z_20240116T000000Z_S1A_30_v1.0", has_vh=False)
        complete = _fake_item("OPERA_L2_RTC-S1_T001-B-IW1_20240115T000000Z_20240116T000000Z_S1A_30_v1.0")

        fake_catalog = MagicMock()
        fake_catalog.search.return_value.items.return_value = [missing_vh, complete]

        src = OPERASource()
        src._catalog = fake_catalog

        import datetime
        passes = src.search_sentinel1(
            bbox={"type": "Polygon", "coordinates": []},
            start_date=datetime.date(2024, 1, 1),
            end_date=datetime.date(2024, 1, 31),
        )
        assert len(passes) == 1
        assert len(passes[0].bursts) == 1
        assert passes[0].bursts[0].id == complete.id


class TestOperaPass:
    def test_geometry_is_bbox_union_of_bursts(self):
        from shapely.geometry import box, mapping

        burst1 = MagicMock()
        burst1.geometry = mapping(box(0, 0, 1, 1))
        burst1.id = "b1"
        burst2 = MagicMock()
        burst2.geometry = mapping(box(2, 2, 3, 3))
        burst2.id = "b2"

        p = OperaPass(pass_id="OPERA_PASS_20240115", bursts=[burst1, burst2])

        assert p.id == "OPERA_PASS_20240115"
        coords = p.geometry["coordinates"][0]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        assert min(xs) == 0 and max(xs) == 3
        assert min(ys) == 0 and max(ys) == 3
