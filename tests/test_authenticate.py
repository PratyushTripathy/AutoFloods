# tests/test_authenticate.py

"""
Tests for autofloods.authenticate (deprecated standalone sign_in(),
superseded by autofloods.sources.MPCSource but kept importable for
backward compatibility) -- all network calls mocked, no live
credentials required.

This module has the same shape of code as autofloods.sources.mpc's
authenticate(), which shipped a real bug for multiple releases:
pystac_client.Client.open() was called with an unsupported timeout=
kwarg against the pinned pystac-client==0.6.1 (see tests/test_sources.py).
These tests assert the exact call signature made to Client.open() here
too, so the same class of bug would fail loudly rather than shipping
undetected a third time.
"""

from unittest.mock import MagicMock, patch

from autofloods.authenticate import sign_in


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
