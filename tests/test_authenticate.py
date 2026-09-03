# tests/test_authenticate.py

"""
Tests for autofloods.authenticate: the deprecated standalone sign_in()
(superseded by autofloods.sources.MPCSource but kept importable for
backward compatibility) and setup_earthdata_login() (the recommended
way to configure NASA Earthdata Login credentials for OPERASource) --
all network calls mocked, no live credentials required, no real
~/.netrc touched (tests always pass an explicit netrc_path=tmp_path).

This module has the same shape of code as autofloods.sources.mpc's
authenticate(), which shipped a real bug for multiple releases:
pystac_client.Client.open() was called with an unsupported timeout=
kwarg against the pinned pystac-client==0.6.1 (see tests/test_sources.py).
These tests assert the exact call signature made to Client.open() here
too, so the same class of bug would fail loudly rather than shipping
undetected a third time.
"""

from unittest.mock import MagicMock, patch

from autofloods.authenticate import sign_in, setup_earthdata_login


class TestSignInAuthenticate:
    def test_sign_in_calls_client_open_with_only_supported_kwargs(self):
        """
        Regression-style test for the shipped bug (see module docstring
        and tests/test_sources.py). autospec binds the mock's spec to
        the real (pinned) Client.open signature, so passing an
        unsupported kwarg like timeout= would raise TypeError here,
        exactly like it did against the real pinned dependency.
        """
        with patch("autofloods.authenticate.pystac_client.Client.open",
                   autospec=True) as mock_open:
            sign_in()

        assert mock_open.called
        _, kwargs = mock_open.call_args
        assert "timeout" not in kwargs

    def test_sign_in_sets_subscription_key_when_env_var_present(self):
        with patch("autofloods.authenticate.pystac_client.Client.open") as mock_open, \
             patch("planetary_computer.settings.set_subscription_key") as mock_set_key, \
             patch.dict("os.environ", {"MPC_SUBSCRIPTION_KEY": "env-key"}, clear=True):
            sign_in()

        mock_set_key.assert_called_once_with("env-key")
        assert mock_open.called

    def test_sign_in_proceeds_anonymously_without_env_var(self):
        with patch("autofloods.authenticate.pystac_client.Client.open") as mock_open, \
             patch("planetary_computer.settings.set_subscription_key") as mock_set_key, \
             patch.dict("os.environ", {}, clear=True):
            sign_in()

        mock_set_key.assert_not_called()
        assert mock_open.called

    def test_sign_in_returns_catalog(self):
        fake_catalog = MagicMock()
        with patch("autofloods.authenticate.pystac_client.Client.open", return_value=fake_catalog):
            result = sign_in()
        assert result is fake_catalog


class TestSetupEarthdataLogin:
    def test_prompts_when_credentials_not_given(self, tmp_path):
        netrc_path = tmp_path / ".netrc"
        with patch("autofloods.authenticate.input", return_value="alice", create=True) as mock_input, \
             patch("autofloods.authenticate.getpass.getpass", return_value="s3cret") as mock_getpass:
            setup_earthdata_login(netrc_path=str(netrc_path))

        mock_input.assert_called_once()
        mock_getpass.assert_called_once()
        text = netrc_path.read_text()
        assert "machine urs.earthdata.nasa.gov" in text
        assert "login alice" in text
        assert "password s3cret" in text

    def test_skips_prompt_when_credentials_given_directly(self, tmp_path):
        netrc_path = tmp_path / ".netrc"
        with patch("autofloods.authenticate.input", create=True) as mock_input, \
             patch("autofloods.authenticate.getpass.getpass") as mock_getpass:
            setup_earthdata_login(username="bob", password="hunter2", netrc_path=str(netrc_path))

        mock_input.assert_not_called()
        mock_getpass.assert_not_called()
        text = netrc_path.read_text()
        assert "login bob" in text
        assert "password hunter2" in text

    def test_preserves_unrelated_existing_entries(self, tmp_path):
        netrc_path = tmp_path / ".netrc"
        netrc_path.write_text("machine example.com\n  login someone\n  password pw\n")

        setup_earthdata_login(username="bob", password="hunter2", netrc_path=str(netrc_path))

        text = netrc_path.read_text()
        assert "machine example.com" in text
        assert "login someone" in text
        assert "machine urs.earthdata.nasa.gov" in text
        assert "login bob" in text

    def test_replaces_existing_earthdata_block_in_place(self, tmp_path):
        netrc_path = tmp_path / ".netrc"
        netrc_path.write_text(
            "machine urs.earthdata.nasa.gov\n  login old_user\n  password old_pw\n"
        )

        setup_earthdata_login(username="new_user", password="new_pw", netrc_path=str(netrc_path))

        text = netrc_path.read_text()
        assert text.count("machine urs.earthdata.nasa.gov") == 1
        assert "old_user" not in text
        assert "old_pw" not in text
        assert "login new_user" in text
        assert "password new_pw" in text

    def test_sets_permissions_to_owner_only_on_posix(self, tmp_path):
        netrc_path = tmp_path / ".netrc"
        with patch("autofloods.authenticate.platform.system", return_value="Linux"), \
             patch("autofloods.authenticate.os.chmod") as mock_chmod:
            setup_earthdata_login(username="bob", password="hunter2", netrc_path=str(netrc_path))

        mock_chmod.assert_called_once_with(str(netrc_path), 0o600)

    def test_skips_chmod_on_windows(self, tmp_path):
        netrc_path = tmp_path / ".netrc"
        with patch("autofloods.authenticate.platform.system", return_value="Windows"), \
             patch("autofloods.authenticate.os.chmod") as mock_chmod:
            setup_earthdata_login(username="bob", password="hunter2", netrc_path=str(netrc_path))

        mock_chmod.assert_not_called()

    def test_returns_netrc_path(self, tmp_path):
        netrc_path = tmp_path / ".netrc"
        result = setup_earthdata_login(username="bob", password="hunter2", netrc_path=str(netrc_path))
        assert result == str(netrc_path)

    def test_verify_failure_raises_runtime_error(self, tmp_path):
        netrc_path = tmp_path / ".netrc"
        with patch("autofloods.authenticate._read_netrc_text", return_value=""):
            try:
                setup_earthdata_login(username="bob", password="hunter2", netrc_path=str(netrc_path))
            except RuntimeError:
                pass
            else:
                raise AssertionError("expected RuntimeError when verification fails")

    def test_no_verify_call_skips_check(self, tmp_path):
        netrc_path = tmp_path / ".netrc"
        with patch("autofloods.authenticate._read_netrc_text", return_value=""):
            result = setup_earthdata_login(
                username="bob", password="hunter2", netrc_path=str(netrc_path), verify=False
            )
        assert result == str(netrc_path)
