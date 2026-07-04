//! # npls1
//!
//! N-PLS1 (N-way Partial Least Squares) regression algorithm in pure Rust.
//!
//! Faithful port of the Python `npls` class. All linear algebra (SVD, matrix
//! inverse, Kronecker product) is implemented from scratch — **no** C, C++,
//! LAPACK, or Fortran dependencies are required.

use ndarray::{s, Array1, Array2, Array3};
use std::collections::HashMap;
use thiserror::Error;

// ═══════════════════════════════════════════════════════════════════
// Errors
// ═══════════════════════════════════════════════════════════════════

#[derive(Error, Debug)]
pub enum NplsError {
    #[error("Linear algebra error: {0}")]
    LinalgError(String),
    #[error("{0}")]
    ValueError(String),
    #[error("Model has not been fitted yet. Call fit() before predict().")]
    NotFitted,
}

// ═══════════════════════════════════════════════════════════════════
// SNR types
// ═══════════════════════════════════════════════════════════════════

pub type SnrResponse = HashMap<String, Vec<f64>>;

#[derive(Debug, Clone)]
pub struct SmoothLoadingResult {
    pub emission: SnrResponse,
    pub excitation: SnrResponse,
}

// ═══════════════════════════════════════════════════════════════════
// Pure-Rust linear algebra helpers
// ═══════════════════════════════════════════════════════════════════

/// Matrix multiply: C = A × B.
fn mat_mul(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (m, k, n) = (a.nrows(), a.ncols(), b.ncols());
    assert_eq!(k, b.nrows(), "mat_mul: incompatible shapes");
    let mut c = Array2::<f64>::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[[i, p]] * b[[p, j]];
            }
            c[[i, j]] = sum;
        }
    }
    c
}

/// Identity matrix n×n.
fn eye(n: usize) -> Array2<f64> {
    let mut m = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        m[[i, i]] = 1.0;
    }
    m
}

/// Transpose of a 2-D array.
fn transpose(a: &Array2<f64>) -> Array2<f64> {
    let (m, n) = (a.nrows(), a.ncols());
    let mut t = Array2::<f64>::zeros((n, m));
    for i in 0..m {
        for j in 0..n {
            t[[j, i]] = a[[i, j]];
        }
    }
    t
}

/// Kronecker product (matches `numpy.kron`).
fn kron(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (ma, na) = (a.nrows(), a.ncols());
    let (mb, nb) = (b.nrows(), b.ncols());
    let mut out = Array2::<f64>::zeros((ma * mb, na * nb));
    for i in 0..ma {
        for j in 0..na {
            let aij = a[[i, j]];
            for p in 0..mb {
                for q in 0..nb {
                    out[[i * mb + p, j * nb + q]] = aij * b[[p, q]];
                }
            }
        }
    }
    out
}

/// Matrix inverse via Gauss-Jordan elimination with partial pivoting.
fn mat_inv(a: &Array2<f64>) -> Result<Array2<f64>, NplsError> {
    let n = a.nrows();
    assert_eq!(n, a.ncols(), "mat_inv: matrix must be square");

    let mut aug = Array2::<f64>::zeros((n, 2 * n));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n + i]] = 1.0;
    }

    for col in 0..n {
        let mut max_val = aug[[col, col]].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let v = aug[[row, col]].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-14 {
            return Err(NplsError::LinalgError(
                "Matrix is singular or nearly singular".to_string(),
            ));
        }
        if max_row != col {
            for j in 0..(2 * n) {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
        }
        let pivot = aug[[col, col]];
        for j in 0..(2 * n) {
            aug[[col, j]] /= pivot;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[[row, col]];
            for j in 0..(2 * n) {
                aug[[row, j]] -= factor * aug[[col, j]];
            }
        }
    }

    let mut inv = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = aug[[i, n + j]];
        }
    }
    Ok(inv)
}

/// Jacobi eigenvalue algorithm for a real symmetric matrix.
/// Returns `(eigenvalues, eigenvectors)` sorted by descending eigenvalue.
fn symmetric_eigen(a: &Array2<f64>) -> Result<(Array1<f64>, Array2<f64>), NplsError> {
    let n = a.nrows();
    assert_eq!(n, a.ncols(), "symmetric_eigen: matrix must be square");

    if n == 0 {
        return Ok((Array1::zeros(0), Array2::zeros((0, 0))));
    }
    if n == 1 {
        let mut v = Array2::zeros((1, 1));
        v[[0, 0]] = 1.0;
        let mut ev = Array1::zeros(1);
        ev[0] = a[[0, 0]];
        return Ok((ev, v));
    }

    let mut aa = a.clone();
    let mut v = eye(n);
    let max_iter = 100 * n * n;
    let tol = 1e-14;

    for _ in 0..max_iter {
        let mut max_off = 0.0_f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let val = aa[[i, j]].abs();
                if val > max_off {
                    max_off = val;
                    p = i;
                    q = j;
                }
            }
        }
        if max_off < tol {
            break;
        }

        let app = aa[[p, p]];
        let aqq = aa[[q, q]];
        let apq = aa[[p, q]];
        let theta = if (app - aqq).abs() < 1e-30 {
            std::f64::consts::FRAC_PI_4 * if apq >= 0.0 { 1.0 } else { -1.0 }
        } else {
            0.5 * (2.0 * apq / (app - aqq)).atan()
        };
        let c = theta.cos();
        let sn = theta.sin();

        let mut new_a = aa.clone();
        for i in 0..n {
            if i != p && i != q {
                let aip = aa[[i, p]];
                let aiq = aa[[i, q]];
                new_a[[i, p]] = c * aip + sn * aiq;
                new_a[[p, i]] = new_a[[i, p]];
                new_a[[i, q]] = -sn * aip + c * aiq;
                new_a[[q, i]] = new_a[[i, q]];
            }
        }
        new_a[[p, p]] = c * c * app + 2.0 * c * sn * apq + sn * sn * aqq;
        new_a[[q, q]] = sn * sn * app - 2.0 * c * sn * apq + c * c * aqq;
        new_a[[p, q]] = 0.0;
        new_a[[q, p]] = 0.0;
        aa = new_a;

        let mut new_v = v.clone();
        for i in 0..n {
            let vip = v[[i, p]];
            let viq = v[[i, q]];
            new_v[[i, p]] = c * vip + sn * viq;
            new_v[[i, q]] = -sn * vip + c * viq;
        }
        v = new_v;
    }

    let mut eigenvalues = Array1::<f64>::zeros(n);
    for i in 0..n {
        eigenvalues[i] = aa[[i, i]];
    }

    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&x, &y| {
        eigenvalues[y]
            .partial_cmp(&eigenvalues[x])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut sv = Array1::<f64>::zeros(n);
    let mut svec = Array2::<f64>::zeros((n, n));
    for (new_i, &old_i) in idx.iter().enumerate() {
        sv[new_i] = eigenvalues[old_i];
        for row in 0..n {
            svec[[row, new_i]] = v[[row, old_i]];
        }
    }
    Ok((sv, svec))
}

/// Full SVD: A = U Σ Vᵀ. Returns `(U (m×m), singular_values, Vt (n×n))`.
/// The sign convention is normalised so `U[:,i]` has its largest-magnitude
/// entry positive — this makes results deterministic and matches the way the
/// dominant singular vectors are consumed by the N-PLS algorithm.
fn svd_full(a: &Array2<f64>) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>), NplsError> {
    let m = a.nrows();
    let n = a.ncols();
    let k = m.min(n);

    let at = transpose(a);

    // Right singular vectors from AᵀA (n×n)
    let ata = mat_mul(&at, a);
    let (eig_v, vmat) = symmetric_eigen(&ata)?;

    let mut sigma = Array1::<f64>::zeros(k);
    for i in 0..k {
        sigma[i] = eig_v[i].max(0.0).sqrt();
    }

    // V is (n×n); columns are right singular vectors.
    // Build U column-by-column: u_i = A v_i / sigma_i.
    let mut u = Array2::<f64>::zeros((m, m));
    let av = mat_mul(a, &vmat); // (m×n)
    for j in 0..k {
        if sigma[j] > 1e-14 {
            for i in 0..m {
                u[[i, j]] = av[[i, j]] / sigma[j];
            }
        }
    }

    // Complete any missing U columns (zero singular values) via AAᵀ eigenvectors.
    let need_u_completion = (0..k).any(|j| sigma[j] <= 1e-14) || m > k;
    if need_u_completion {
        let aat = mat_mul(a, &at); // (m×m)
        let (_eig_u, umat) = symmetric_eigen(&aat)?;
        // Fill columns k..m (and any zero-sigma columns) from umat, keeping
        // the already-computed reliable columns.
        for j in 0..m {
            let reliable = j < k && sigma[j] > 1e-14;
            if !reliable {
                for i in 0..m {
                    u[[i, j]] = umat[[i, j]];
                }
            }
        }
    }

    // Vt = Vᵀ
    let vt = transpose(&vmat);

    // Sign normalisation: make the largest-magnitude entry of each U column
    // positive, and flip the corresponding Vt row to keep A = U Σ Vᵀ.
    let mut u_final = u;
    let mut vt_final = vt;
    for j in 0..k {
        // find largest magnitude entry in column j of U
        let mut best = 0usize;
        let mut best_val = 0.0_f64;
        for i in 0..m {
            let v = u_final[[i, j]].abs();
            if v > best_val {
                best_val = v;
                best = i;
            }
        }
        if u_final[[best, j]] < 0.0 {
            for i in 0..m {
                u_final[[i, j]] = -u_final[[i, j]];
            }
            for col in 0..n {
                vt_final[[j, col]] = -vt_final[[j, col]];
            }
        }
    }

    Ok((u_final, sigma, vt_final))
}

// ═══════════════════════════════════════════════════════════════════
// Npls struct
// ═══════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct Npls {
    // configuration
    pub n_components: usize,
    pub a: f64,
    pub derivative_rang: Vec<usize>,
    pub norm_func: Vec<String>,
    pub crash_norm_name: Option<String>,
    pub crash_norm_value: Option<f64>,
    pub excitation_wavelenth: Array1<f64>,
    pub emission_wavelenth: Array1<f64>,

    // fitted state
    pub w_k: Option<Vec<Array2<f64>>>,
    pub w_i: Option<Vec<Array2<f64>>>,
    pub bf_array: Option<Vec<Array2<f64>>>,
    pub train_error: Option<f64>,
    pub snr_emission: Option<Vec<SnrResponse>>,
    pub snr_excitation: Option<Vec<SnrResponse>>,
}

impl Npls {
    // ─────────────────────────── Constructors ───────────────────────

    pub fn new(n_components: usize, a: f64) -> Self {
        Self {
            n_components,
            a,
            derivative_rang: vec![],
            norm_func: vec![],
            crash_norm_name: None,
            crash_norm_value: None,
            excitation_wavelenth: Array1::zeros(1),
            emission_wavelenth: Array1::zeros(1),
            w_k: None,
            w_i: None,
            bf_array: None,
            train_error: None,
            snr_emission: None,
            snr_excitation: None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_snr(
        n_components: usize,
        a: f64,
        derivative_rang: Vec<usize>,
        norm_func: Vec<String>,
        crash_norm_name: Option<String>,
        crash_norm_value: Option<f64>,
        excitation_wavelenth: Array1<f64>,
        emission_wavelenth: Array1<f64>,
    ) -> Self {
        Self {
            n_components,
            a,
            derivative_rang,
            norm_func,
            crash_norm_name,
            crash_norm_value,
            excitation_wavelenth,
            emission_wavelenth,
            w_k: None,
            w_i: None,
            bf_array: None,
            train_error: None,
            snr_emission: None,
            snr_excitation: None,
        }
    }

    // ─────────────────────────── SNR helpers ────────────────────────

    fn chek_smooth_one_model(signal: &[f64], x: &[f64]) -> SnrResponse {
        let mut response: SnrResponse = HashMap::new();
        if signal.len() < 2 || x.len() < 2 {
            return response;
        }
        let n = signal.len();
        let mut derivative = vec![0.0f64; n - 1];
        for i in 0..n - 1 {
            let dx = x[i + 1] - x[i];
            derivative[i] = if dx.abs() < 1e-30 {
                0.0
            } else {
                (signal[i + 1] - signal[i]) / dx
            };
        }
        let signal_l2 = signal.iter().map(|v| v * v).sum::<f64>().sqrt();
        let noise_l2 = derivative.iter().map(|v| v * v).sum::<f64>().sqrt();
        let evklid = if noise_l2 > 1e-30 { signal_l2 / noise_l2 } else { f64::INFINITY };
        response.entry("evklid".into()).or_default().push(evklid);

        let s_max = signal.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        let n_max = derivative.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        let max_snr = if n_max > 1e-30 { s_max / n_max } else { f64::INFINITY };
        response.entry("max".into()).or_default().push(max_snr);

        let s_mean = signal.iter().map(|v| v.abs()).sum::<f64>() / n as f64;
        let n_mean = derivative.iter().map(|v| v.abs()).sum::<f64>() / derivative.len() as f64;
        let mean_snr = if n_mean > 1e-30 { s_mean / n_mean } else { f64::INFINITY };
        response.entry("mean".into()).or_default().push(mean_snr);

        response
    }

    fn check_smooth_loadings(
        &self,
        w_i: &[f64],
        w_k: &[f64],
        n_component: usize,
    ) -> Result<SmoothLoadingResult, NplsError> {
        let resp_emission =
            Self::chek_smooth_one_model(w_k, self.emission_wavelenth.as_slice().unwrap());
        let resp_excitation =
            Self::chek_smooth_one_model(w_i, self.excitation_wavelenth.as_slice().unwrap());

        if let Some(ref norm_name) = self.crash_norm_name {
            let crash_value = self.crash_norm_value.unwrap_or(0.0);
            if let Some(vals) = resp_emission.get(norm_name) {
                for v in vals {
                    if *v <= crash_value {
                        return Err(NplsError::ValueError(format!(
                            "Emission {} component is a very noisy. May be you can choose another norm. Now you choose {} norm",
                            n_component, norm_name
                        )));
                    }
                }
            }
            if let Some(vals) = resp_excitation.get(norm_name) {
                for v in vals {
                    if *v <= crash_value {
                        return Err(NplsError::ValueError(format!(
                            "Excitation {} component is a very noisy. May be you can choose another norm. Now you choose {} norm",
                            n_component, norm_name
                        )));
                    }
                }
            }
        }

        Ok(SmoothLoadingResult {
            emission: resp_emission,
            excitation: resp_excitation,
        })
    }

    // ─────────────────────────────── fit ────────────────────────────

    /// Fit the N-PLS1 model.
    ///
    /// * `xtrain` — shape `(n_samples, n_emission, n_excitation)`
    /// * `ytrain` — shape `(n_samples,)`
    pub fn fit(
        &mut self,
        xtrain: &Array3<f64>,
        ytrain: &Array1<f64>,
    ) -> Result<&mut Self, NplsError> {
        let n0 = xtrain.shape()[0]; // x.shape[0]  (n_samples)
        let n1 = xtrain.shape()[1]; // x.shape[1]  (emission)
        let n2 = xtrain.shape()[2]; // x.shape[2]  (excitation)

        let mut x = xtrain.clone();
        let mut y = ytrain.clone();
        let y_copy = ytrain.clone();

        let do_snr = !self.derivative_rang.is_empty();
        let mut snr_emission: Vec<SnrResponse> = Vec::new();
        let mut snr_excitation: Vec<SnrResponse> = Vec::new();

        // Tt : (n0 × n_components)
        let mut tt = Array2::<f64>::zeros((n0, self.n_components));
        // mass : accumulated fitted values
        let mut mass = Array1::<f64>::zeros(n0);

        let mut w_k_mass: Vec<Array2<f64>> = Vec::with_capacity(self.n_components);
        let mut w_i_mass: Vec<Array2<f64>> = Vec::with_capacity(self.n_components);
        let mut bf_array: Vec<Array2<f64>> = Vec::with_capacity(self.n_components);

        let mut mmas = Array3::<f64>::zeros((n0, n1, n2));

        for f in 0..self.n_components {
            // z = Σ_i y[i] * x[i,:,:]   (n1 × n2)
            let mut z = Array2::<f64>::zeros((n1, n2));
            for i in 0..n0 {
                let xi = x.slice(s![i, .., ..]);
                for r in 0..n1 {
                    for c in 0..n2 {
                        z[[r, c]] += y[i] * xi[[r, c]];
                    }
                }
            }

            // SVD: Wk, S, WI = svd(z)
            let (wk_mat, _s, wi_mat) = svd_full(&z)?;

            // w_k = Wk[:,0] -> (n1 × 1)
            let mut w_k = Array2::<f64>::zeros((n1, 1));
            for r in 0..n1 {
                w_k[[r, 0]] = wk_mat[[r, 0]];
            }
            // w_i = WI[0,:] -> (n2 × 1)
            let mut w_i = Array2::<f64>::zeros((n2, 1));
            for c in 0..n2 {
                w_i[[c, 0]] = wi_mat[[0, c]];
            }

            // SNR check
            if do_snr {
                let w_i_flat: Vec<f64> = (0..n2).map(|c| w_i[[c, 0]]).collect();
                let w_k_flat: Vec<f64> = (0..n1).map(|r| w_k[[r, 0]]).collect();
                let response = self.check_smooth_loadings(&w_i_flat, &w_k_flat, f)?;
                snr_emission.push(response.emission);
                snr_excitation.push(response.excitation);
            }

            w_k_mass.push(w_k.clone());
            w_i_mass.push(w_i.clone());

            // Tt[h,f] = w_i^T @ x[h]^T @ w_k
            for h in 0..n0 {
                let xh = x.slice(s![h, .., ..]); // (n1 × n2)
                // t1 = x[h]^T @ w_k  -> (n2 × 1)
                let mut t1 = Array1::<f64>::zeros(n2);
                for c in 0..n2 {
                    let mut acc = 0.0;
                    for r in 0..n1 {
                        acc += xh[[r, c]] * w_k[[r, 0]];
                    }
                    t1[c] = acc;
                }
                // val = w_i^T @ t1
                let mut val = 0.0;
                for c in 0..n2 {
                    val += w_i[[c, 0]] * t1[c];
                }
                tt[[h, f]] = val;
            }

            // T = Tt[:, 0:f+1]  -> (n0 × (f+1))
            let fp1 = f + 1;
            let mut t_mat = Array2::<f64>::zeros((n0, fp1));
            for h in 0..n0 {
                for col in 0..fp1 {
                    t_mat[[h, col]] = tt[[h, col]];
                }
            }

            // bf = ((inv(T @ T^T - a*I) @ T)^T) @ y   -> ((f+1) × 1)
            let t_t = transpose(&t_mat); // ((f+1) × n0)
            let ttt = mat_mul(&t_mat, &t_t); // (n0 × n0)
            let a_eye = eye(n0);
            let mut inner = Array2::<f64>::zeros((n0, n0));
            for i in 0..n0 {
                for j in 0..n0 {
                    inner[[i, j]] = ttt[[i, j]] - self.a * a_eye[[i, j]];
                }
            }
            let inner_inv = mat_inv(&inner)?; // (n0 × n0)
            let tmp = mat_mul(&inner_inv, &t_mat); // (n0 × (f+1))
            let tmp_t = transpose(&tmp); // ((f+1) × n0)
            let mut y_col = Array2::<f64>::zeros((n0, 1));
            for h in 0..n0 {
                y_col[[h, 0]] = y[h];
            }
            let bf = mat_mul(&tmp_t, &y_col); // ((f+1) × 1)
            bf_array.push(bf.clone());

            // WW = kron(w_k, w_i).reshape(n1, n2)
            let ww_kron = kron(&w_k, &w_i); // ((n1*n2) × 1)
            let mut ww = Array2::<f64>::zeros((n1, n2));
            {
                let flat: Vec<f64> = ww_kron.iter().copied().collect();
                let mut idx = 0;
                for r in 0..n1 {
                    for c in 0..n2 {
                        ww[[r, c]] = flat[idx];
                        idx += 1;
                    }
                }
            }

            // mmas[g,:,:] = Tt[g,f] * WW ; then x = x - mmas
            for g in 0..n0 {
                let scale = tt[[g, f]];
                for r in 0..n1 {
                    for c in 0..n2 {
                        mmas[[g, r, c]] = scale * ww[[r, c]];
                    }
                }
            }
            for g in 0..n0 {
                for r in 0..n1 {
                    for c in 0..n2 {
                        x[[g, r, c]] -= mmas[[g, r, c]];
                    }
                }
            }

            // T @ bf  -> (n0 × 1)
            let t_bf = mat_mul(&t_mat, &bf); // (n0 × 1)
            // y = y - (T @ bf)
            for h in 0..n0 {
                y[h] -= t_bf[[h, 0]];
            }
            // mass += (T @ bf)
            for h in 0..n0 {
                mass[h] += t_bf[[h, 0]];
            }
        }

        // train_error = mean((mass - y_copy)^2)
        let mut sq_sum = 0.0;
        for h in 0..n0 {
            let d = mass[h] - y_copy[h];
            sq_sum += d * d;
        }
        let train_error = sq_sum / n0 as f64;

        self.bf_array = Some(bf_array);
        self.train_error = Some(train_error);
        self.w_k = Some(w_k_mass);
        self.w_i = Some(w_i_mass);
        if do_snr {
            self.snr_emission = Some(snr_emission);
            self.snr_excitation = Some(snr_excitation);
        }

        Ok(self)
    }

    // ───────────────────────────── predict ──────────────────────────

    /// Predict target values for `xtest` (shape `(n_samples, n_emission, n_excitation)`).
    pub fn predict(&self, xtest: &Array3<f64>) -> Result<Array1<f64>, NplsError> {
        let w_k = self.w_k.as_ref().ok_or(NplsError::NotFitted)?;
        let w_i = self.w_i.as_ref().ok_or(NplsError::NotFitted)?;
        let bf_array = self.bf_array.as_ref().ok_or(NplsError::NotFitted)?;

        let n0 = xtest.shape()[0];
        let n1 = xtest.shape()[1];
        let n2 = xtest.shape()[2];

        let mut x = xtest.clone();
        let mut tt = Array2::<f64>::zeros((n0, self.n_components));
        let mut y = Array1::<f64>::zeros(n0);
        let mut mmas = Array3::<f64>::zeros((n0, n1, n2));

        for f in 0..self.n_components {
            let w_k_f = &w_k[f]; // (n1 × 1)
            let w_i_f = &w_i[f]; // (n2 × 1)

            // Tt[h,f] = w_i^T @ x[h]^T @ w_k
            for h in 0..n0 {
                let xh = x.slice(s![h, .., ..]);
                let mut t1 = Array1::<f64>::zeros(n2);
                for c in 0..n2 {
                    let mut acc = 0.0;
                    for r in 0..n1 {
                        acc += xh[[r, c]] * w_k_f[[r, 0]];
                    }
                    t1[c] = acc;
                }
                let mut val = 0.0;
                for c in 0..n2 {
                    val += w_i_f[[c, 0]] * t1[c];
                }
                tt[[h, f]] = val;
            }

            // T = Tt[:, 0:f+1]
            let fp1 = f + 1;
            let mut t_mat = Array2::<f64>::zeros((n0, fp1));
            for h in 0..n0 {
                for col in 0..fp1 {
                    t_mat[[h, col]] = tt[[h, col]];
                }
            }

            // WW = kron(w_k, w_i).reshape(n1, n2)
            let ww_kron = kron(w_k_f, w_i_f);
            let mut ww = Array2::<f64>::zeros((n1, n2));
            {
                let flat: Vec<f64> = ww_kron.iter().copied().collect();
                let mut idx = 0;
                for r in 0..n1 {
                    for c in 0..n2 {
                        ww[[r, c]] = flat[idx];
                        idx += 1;
                    }
                }
            }

            // mmas[g,:,:] = Tt[g,f] * WW ; x = x - mmas
            for g in 0..n0 {
                let scale = tt[[g, f]];
                for r in 0..n1 {
                    for c in 0..n2 {
                        mmas[[g, r, c]] = scale * ww[[r, c]];
                    }
                }
            }
            for g in 0..n0 {
                for r in 0..n1 {
                    for c in 0..n2 {
                        x[[g, r, c]] -= mmas[[g, r, c]];
                    }
                }
            }

            // y = y + (T @ bf_array[f])
            let t_bf = mat_mul(&t_mat, &bf_array[f]); // (n0 × 1)
            for h in 0..n0 {
                y[h] += t_bf[[h, 0]];
            }
        }

        Ok(y)
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn make_data() -> (Array3<f64>, Array1<f64>) {
        let mut x = Array3::<f64>::zeros((5, 4, 3));
        for i in 0..5 {
            for j in 0..4 {
                for k in 0..3 {
                    x[[i, j, k]] = ((i + 1) * (j + 1) * (k + 1)) as f64
                        + 0.1 * (i as f64 - j as f64 + k as f64);
                }
            }
        }
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        (x, y)
    }

    #[test]
    fn test_kron_matches_numpy() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[0.0, 5.0], [6.0, 7.0]];
        let k = kron(&a, &b);
        // numpy.kron reference
        let expected = array![
            [0.0, 5.0, 0.0, 10.0],
            [6.0, 7.0, 12.0, 14.0],
            [0.0, 15.0, 0.0, 20.0],
            [18.0, 21.0, 24.0, 28.0]
        ];
        for i in 0..4 {
            for j in 0..4 {
                assert!((k[[i, j]] - expected[[i, j]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_inv_identity() {
        let a = array![[4.0, 7.0], [2.0, 6.0]];
        let inv = mat_inv(&a).unwrap();
        let prod = mat_mul(&a, &inv);
        assert!((prod[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((prod[[1, 1]] - 1.0).abs() < 1e-10);
        assert!(prod[[0, 1]].abs() < 1e-10);
        assert!(prod[[1, 0]].abs() < 1e-10);
    }

    #[test]
    fn test_svd_reconstruction() {
        let a = array![[3.0, 1.0, 1.0], [-1.0, 3.0, 1.0]];
        let (u, s, vt) = svd_full(&a).unwrap();
        // Reconstruct: A ≈ U[:, :k] Σ Vt[:k, :]
        let (m, n) = (a.nrows(), a.ncols());
        let k = m.min(n);
        let mut recon = Array2::<f64>::zeros((m, n));
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0;
                for r in 0..k {
                    acc += u[[i, r]] * s[r] * vt[[r, j]];
                }
                recon[[i, j]] = acc;
            }
        }
        for i in 0..m {
            for j in 0..n {
                assert!(
                    (recon[[i, j]] - a[[i, j]]).abs() < 1e-8,
                    "svd reconstruction mismatch at [{},{}]",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_fit_predict_runs() {
        let (x, y) = make_data();
        let mut model = Npls::new(2, 3.0);
        model.fit(&x, &y).unwrap();
        assert!(model.train_error.is_some());
        assert!(model.w_k.is_some());
        assert!(model.w_i.is_some());
        assert!(model.bf_array.is_some());

        let preds = model.predict(&x).unwrap();
        assert_eq!(preds.len(), 5);
        for v in preds.iter() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_predict_before_fit_errors() {
        let (x, _) = make_data();
        let model = Npls::new(2, 3.0);
        let res = model.predict(&x);
        assert!(matches!(res, Err(NplsError::NotFitted)));
    }

    #[test]
    fn test_with_snr_constructor() {
        let (x, y) = make_data();
        let mut model = Npls::with_snr(
            2,
            3.0,
            vec![1],
            vec![],
            Some("evklid".to_string()),
            Some(-1.0), // very low threshold => won't crash
            array![1.0, 2.0, 3.0],       // excitation (n2 = 3)
            array![1.0, 2.0, 3.0, 4.0],  // emission   (n1 = 4)
        );
        model.fit(&x, &y).unwrap();
        assert!(model.snr_emission.is_some());
        assert!(model.snr_excitation.is_some());
    }

    #[test]
    fn test_snr_crash_triggers() {
        let (x, y) = make_data();
        let mut model = Npls::with_snr(
            2,
            3.0,
            vec![1],
            vec![],
            Some("evklid".to_string()),
            Some(1e18), // huge threshold => must crash
            array![1.0, 2.0, 3.0],
            array![1.0, 2.0, 3.0, 4.0],
        );
        let res = model.fit(&x, &y);
        assert!(matches!(res, Err(NplsError::ValueError(_))));
    }
}