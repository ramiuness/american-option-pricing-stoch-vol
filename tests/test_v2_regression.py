"""
Diagnostic & regression tests for amOptPricer and priceModels.

Phase 1: Isolate v0/current differences.
Phase 2: Unit tests for key functions.

Run: python -m pytest tests/test_v2_regression.py -v
"""
import sys
import numpy as np
import pytest

sys.path.insert(0, "src")
import priceModels_v0 as pm_v0
import priceModels as pm
import amOptPricer_v0 as aop_v0
import amerPrice as ap


# ── Fixtures ──────────────────────────────────────────────────────────────

TABLE1 = dict(r=0.01, rho=-0.2, kappa=5, nu=0.2,
              sigma0=0.15, theta0=0.18, lam=0.9, eta=0.01)
TABLE2 = dict(r=0.01, rho=0.1691, kappa=4.9394, nu=0.3943,
              sigma0=0.2924, theta0=0.1319, lam=0.3115, eta=0.4112)

K = 100.0
SEED = 42


def _make_model_sim(params, S0, T, n_steps, n_paths):
    """Create v1 model + simulation (shared by both v1/v2 pricers)."""
    m = pm_v0.ImprovedSteinStein(**params, seed=SEED)
    sim = m.simulate_prices(S0=S0, T=T, n_steps_mc=n_steps, n_paths=n_paths)
    return m, sim


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: v1/v2 agreement tests
# ═══════════════════════════════════════════════════════════════════════════

class TestV1V2CVModeAgreement:
    """When use_cv=True with ridge>0, v1 and v2 must produce identical results."""

    @pytest.mark.parametrize("S0", [85.0, 90.0, 100.0, 110.0, 115.0])
    @pytest.mark.parametrize("T,n_steps", [(1/12, 22), (1.0, 52)])
    def test_cv_bs_ridge_identical(self, S0, T, n_steps):
        m, sim = _make_model_sim(TABLE1, S0, T, n_steps, 3000)
        r1 = aop_v0.price_american_put_lsm_llh(m, sim, K, use_cv=True,
                                                 euro_method='bs', ridge=1e-5)
        r2 = ap.price_american_put_lsm_llh(m, sim, K, use_cv=True,
                                                 euro_method='bs', ridge=1e-5)
        assert r1['price'] == pytest.approx(r2['price'], abs=1e-10)
        assert r1['price_imp'] == pytest.approx(r2['price_imp'], abs=1e-10)
        assert r1['std_err_imp'] == pytest.approx(r2['std_err_imp'], abs=1e-10)

    @pytest.mark.parametrize("S0", [85.0, 100.0, 115.0])
    def test_cv_mc1_ridge_identical(self, S0):
        m, sim = _make_model_sim(TABLE1, S0, 1.0, 52, 3000)
        r1 = aop_v0.price_american_put_lsm_llh(m, sim, K, use_cv=True,
                                                 euro_method='mc1', ridge=1e-5)
        r2 = ap.price_american_put_lsm_llh(m, sim, K, use_cv=True,
                                                 euro_method='mc1', ridge=1e-5)
        assert r1['price'] == pytest.approx(r2['price'], abs=1e-10)
        assert r1['price_imp'] == pytest.approx(r2['price_imp'], abs=1e-10)

    @pytest.mark.parametrize("S0", [90.0, 100.0])
    def test_cv_llh_ridge_identical(self, S0):
        m, sim = _make_model_sim(TABLE1, S0, 1/12, 22, 1000)
        r1 = aop_v0.price_american_put_lsm_llh(m, sim, K, use_cv=True,
                                                 euro_method='llh', ridge=1e-5,
                                                 phi_max=300, n_phi=513, n_steps_rk4=128)
        r2 = ap.price_american_put_lsm_llh(m, sim, K, use_cv=True,
                                                 euro_method='llh', ridge=1e-5,
                                                 phi_max=300, n_phi=513, n_steps_rk4=128)
        # LLH mode: ITM-only optimization introduces small numerical differences
        # for near-ATM paths (Ej=0 for barely-OTM paths at j>1 vs small positive in v1)
        assert r1['price'] == pytest.approx(r2['price'], abs=1e-2)
        assert r1['price_imp'] == pytest.approx(r2['price_imp'], abs=1e-2)

    def test_table2_params_cv_bs(self):
        m, sim = _make_model_sim(TABLE2, 100.0, 1.0, 52, 3000)
        r1 = aop_v0.price_american_put_lsm_llh(m, sim, K, use_cv=True,
                                                 euro_method='bs', ridge=1e-5)
        r2 = ap.price_american_put_lsm_llh(m, sim, K, use_cv=True,
                                                 euro_method='bs', ridge=1e-5)
        assert r1['price_imp'] == pytest.approx(r2['price_imp'], abs=1e-10)


class TestUseCvFalseFloor:
    """use_cv=False uses BS European floor (fast). Prices should be close to v1's LLH floor."""

    @pytest.mark.parametrize("S0", [85.0, 100.0, 115.0])
    @pytest.mark.parametrize("T,n_steps", [(1/12, 22), (1.0, 52)])
    def test_bs_floor_close_to_v1(self, S0, T, n_steps):
        """v2 BS floor should produce prices within SE of v1 LLH floor."""
        m, sim = _make_model_sim(TABLE1, S0, T, n_steps, 5000)
        r1 = aop_v0.price_american_put_lsm_llh(m, sim, K, use_cv=False, ridge=1e-5)
        r2 = ap.price_american_put_lsm_llh(m, sim, K, use_cv=False, ridge=1e-5)
        # BS floor should match LLH floor within 2 standard errors
        tol = 2 * max(r1['std_err'], r2['std_err'])
        assert abs(r1['price'] - r2['price']) < tol, \
            f"BS floor diff too large: v1={r1['price']:.6f} v2={r2['price']:.6f} tol={tol:.6f}"


class TestFloorMethodEquivalence:
    """BS floor produces same prices as LLH floor (within SE)."""

    @pytest.mark.parametrize("S0", [85.0, 100.0, 115.0])
    def test_bs_vs_llh_floor(self, S0):
        m, sim = _make_model_sim(TABLE1, S0, 1/12, 22, 3000)
        r_bs = ap.price_american_put_lsm_llh(
            m, sim, K, use_cv=False, ridge=1e-5, floor_method='bs')
        r_llh = ap.price_american_put_lsm_llh(
            m, sim, K, use_cv=False, ridge=1e-5, floor_method='llh')
        tol = 2 * max(r_bs['std_err'], r_llh['std_err'])
        assert abs(r_bs['price'] - r_llh['price']) < tol


class TestAmericanPutBounds:
    """Basic sanity: American price >= European price (EEP >= 0)."""

    @pytest.mark.parametrize("S0", [85.0, 90.0, 100.0, 110.0, 115.0])
    @pytest.mark.parametrize("T,n_steps", [(1/12, 22), (1.0, 52)])
    def test_eep_nonnegative_cv_bs(self, S0, T, n_steps):
        m, sim = _make_model_sim(TABLE1, S0, T, n_steps, 5000)
        res = ap.price_american_put_lsm_llh(m, sim, K, use_cv=True,
                                                  euro_method='bs', ridge=1e-5)
        am_price = res.get('price_imp', res['price'])

        euro_put = m.price_put_llh(S=S0, K=K, tau=T, vol=m.sigma0, theta=m.theta0,
                                    phi_max=300, n_phi=513, n_steps_ode=128).item()
        eep = am_price - euro_put
        # EEP should be >= 0 (allow small negative for MC noise)
        assert eep >= -3 * res.get('std_err_imp', res['std_err']), \
            f"EEP = {eep:.6f} is too negative for S0={S0}, T={T}"


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Unit tests for key functions
# ═══════════════════════════════════════════════════════════════════════════

class TestOLSFitPredictMulti:
    """Verify _ols_fit_predict_multi matches individual calls."""

    def test_matches_individual_with_ridge(self):
        rng = np.random.default_rng(0)
        n_itm, n_all, p = 80, 200, 4
        Phi = rng.standard_normal((n_itm, p))
        Phi_all = rng.standard_normal((n_all, p))
        ys = [rng.standard_normal(n_itm) for _ in range(4)]

        # Individual calls
        preds_individual = [ap._ols_fit_predict(Phi, y, Phi_all, ridge=1e-5) for y in ys]
        # Multi call
        Y_mat = np.column_stack(ys)
        preds_multi = ap._ols_fit_predict_multi(Phi, Y_mat, Phi_all, ridge=1e-5)

        for i, pred_ind in enumerate(preds_individual):
            np.testing.assert_allclose(preds_multi[:, i], pred_ind, rtol=1e-10,
                                       err_msg=f"Column {i} mismatch")

    def test_matches_individual_ridge_zero(self):
        """With ridge=0, multi uses normal eqs while individual uses lstsq.
        Results should still be close for well-conditioned systems."""
        rng = np.random.default_rng(1)
        n_itm, n_all, p = 200, 500, 4
        Phi = rng.standard_normal((n_itm, p))
        Phi_all = rng.standard_normal((n_all, p))
        ys = [rng.standard_normal(n_itm) for _ in range(4)]

        preds_individual = [ap._ols_fit_predict(Phi, y, Phi_all, ridge=0.0) for y in ys]
        Y_mat = np.column_stack(ys)
        preds_multi = ap._ols_fit_predict_multi(Phi, Y_mat, Phi_all, ridge=0.0)

        for i, pred_ind in enumerate(preds_individual):
            np.testing.assert_allclose(preds_multi[:, i], pred_ind, rtol=1e-6,
                                       err_msg=f"Column {i} mismatch with ridge=0")

    def test_empty_phi_returns_zeros(self):
        Phi = np.empty((0, 4))
        Phi_all = np.ones((100, 4))
        Y_mat = np.empty((0, 3))
        result = ap._ols_fit_predict_multi(Phi, Y_mat, Phi_all)
        assert result.shape == (100, 3)
        np.testing.assert_array_equal(result, 0.0)


class TestLaguerreBasis:
    """Verify Laguerre basis against known values."""

    def test_known_values(self):
        x = np.array([0.0, 0.5, 1.0, 2.0])
        Phi = ap._laguerre_basis(x, order=2)
        # L_0(x) = 1
        np.testing.assert_allclose(Phi[:, 0], 1.0)
        # L_1(x) = 1 - x
        np.testing.assert_allclose(Phi[:, 1], 1 - x)
        # L_2(x) = 1 - 2x + x^2/2
        np.testing.assert_allclose(Phi[:, 2], 1 - 2*x + x**2/2)

    def test_shape(self):
        x = np.linspace(0.5, 1.5, 100)
        Phi = ap._laguerre_basis(x, order=3)
        assert Phi.shape == (100, 4)


# ═══════════════════════════════════════════════════════════════════════════
# priceModels v2 tests
# ═══════════════════════════════════════════════════════════════════════════

class TestPriceModelsV2:
    """Verify priceModels matches v0."""

    def test_simulation_bit_identity(self):
        m1 = pm_v0.ImprovedSteinStein(**TABLE1, seed=SEED)
        m2 = pm.ImprovedSteinStein(**TABLE1, seed=SEED)
        s1 = m1.simulate_prices(100.0, 1.0, 52, 1000)
        s2 = m2.simulate_prices(100.0, 1.0, 52, 1000)
        for key in ['S', 'sigma_hat', 'B', 'W', 'W2']:
            np.testing.assert_array_equal(s1[key], s2[key], err_msg=f"{key} mismatch")

    @pytest.mark.parametrize("S0,K", [(100.0, 90.0), (90.0, 100.0), (100.0, 100.0)])
    def test_european_call_bit_identity(self, S0, K):
        m1 = pm_v0.ImprovedSteinStein(**TABLE1, seed=SEED)
        m2 = pm.ImprovedSteinStein(**TABLE1, seed=SEED)
        c1 = m1.price_call_llh(S=S0, K=K, tau=1.0, vol=0.15, theta=0.18,
                                phi_max=300, n_phi=513, n_steps_ode=128).item()
        c2 = m2.price_call_llh(S=S0, K=K, tau=1.0, vol=0.15, theta=0.18,
                                phi_max=300, n_phi=513, n_steps_ode=128).item()
        assert c1 == c2

    def test_put_call_parity(self):
        m = pm.ImprovedSteinStein(**TABLE1, seed=SEED)
        S0, K_val, tau = 100.0, 90.0, 1.0
        call = m.price_call_llh(S=S0, K=K_val, tau=tau, vol=0.15, theta=0.18,
                                 phi_max=300, n_phi=513, n_steps_ode=128).item()
        put = m.price_put_llh(S=S0, K=K_val, tau=tau, vol=0.15, theta=0.18,
                               phi_max=300, n_phi=513, n_steps_ode=128).item()
        # put + S = call + K*exp(-r*tau)
        lhs = put + S0
        rhs = call + K_val * np.exp(-m.r * tau)
        assert lhs == pytest.approx(rhs, abs=1e-10)

    def test_rk4_zero_tau(self):
        phi = pm.phi_grid(300.0, 513)
        rhs = pm.rhs_factory(phi, 0.01, 5.0, 0.2, 0.9, 0.01, -0.2)
        coeffs = pm.rk4_integrate(rhs, tau=0.0, n_steps_ode=128, n_phi=513)
        for k in "CDEFGH":
            np.testing.assert_array_equal(coeffs[k], 0.0)

    def test_rk4_convergence(self):
        """Higher n_steps_ode should converge."""
        m = pm.ImprovedSteinStein(**TABLE1, seed=SEED)
        pre_64 = m.llh_precompute_tau(1.0, 300.0, 513, 64)
        pre_128 = m.llh_precompute_tau(1.0, 300.0, 513, 128)
        pre_256 = m.llh_precompute_tau(1.0, 300.0, 513, 256)

        c_64 = m.price_call_llh(S=100.0, K=90.0, tau=1.0, vol=0.15, theta=0.18, pre=pre_64).item()
        c_128 = m.price_call_llh(S=100.0, K=90.0, tau=1.0, vol=0.15, theta=0.18, pre=pre_128).item()
        c_256 = m.price_call_llh(S=100.0, K=90.0, tau=1.0, vol=0.15, theta=0.18, pre=pre_256).item()

        # Error should decrease: |c_64 - c_256| > |c_128 - c_256|
        err_64 = abs(c_64 - c_256)
        err_128 = abs(c_128 - c_256)
        assert err_128 < err_64 or err_128 < 1e-12, \
            f"RK4 not converging: err_64={err_64:.2e}, err_128={err_128:.2e}"

    def test_table2_european_call(self):
        m1 = pm_v0.ImprovedSteinStein(**TABLE2, seed=SEED)
        m2 = pm.ImprovedSteinStein(**TABLE2, seed=SEED)
        c1 = m1.price_call_llh(S=100.0, K=100.0, tau=1.0, vol=0.2924, theta=0.1319,
                                phi_max=300, n_phi=513, n_steps_ode=128).item()
        c2 = m2.price_call_llh(S=100.0, K=100.0, tau=1.0, vol=0.2924, theta=0.1319,
                                phi_max=300, n_phi=513, n_steps_ode=128).item()
        assert c1 == c2
