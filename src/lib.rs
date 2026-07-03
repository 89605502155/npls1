use ndarray::{Array1, Array2, Array3, s};
use scirs2_linalg::compat::ArrayLinalgExt;

pub struct NPLS1 {
    pub n_components: usize,
    pub a: f64,
    pub bf_array: Vec<Array2<f64>>,
    pub train_error: f64,
    pub w_k: Vec<Array2<f64>>,
    pub w_i: Vec<Array2<f64>>,
}

impl NPLS1 {
    pub fn new(n_components: usize, a: f64) -> Self {
        Self {
            n_components,
            a,
            bf_array: Vec::new(),
            train_error: 0.0,
            w_k: Vec::new(),
            w_i: Vec::new(),
        }
    }

    pub fn fit(&mut self, xtrain: &Array3<f64>, ytrain: &Array1<f64>) -> Result<(), &'static str> {
        let n_samples = xtrain.shape()[0];
        let j = xtrain.shape()[1];
        let k = xtrain.shape()[2];

        let mut x = xtrain.clone();
        let mut y = ytrain.clone();
        let y_copy = ytrain.clone();

        let mut tt = Array2::zeros((n_samples, self.n_components));
        let mut mass = Array1::zeros(n_samples);

        self.bf_array.clear();
        self.w_k.clear();
        self.w_i.clear();

        for f in 0..self.n_components {
            let mut z = Array2::zeros((j, k));
            for i in 0..n_samples {
                z += &(y[i] * &x.slice(s![i, .., ..]).to_owned());
            }

            let (u, _s, vt) = z.svd(true).map_err(|_| "SVD failed")?;

            let mut w_k = Array2::zeros((j, 1));
            for r in 0..j {
                w_k[(r, 0)] = u[(r, 0)];
            }

            let mut w_i = Array2::zeros((k, 1));
            for c in 0..k {
                w_i[(c, 0)] = vt[(0, c)];
            }

            self.w_k.push(w_k.clone());
            self.w_i.push(w_i.clone());

            for h in 0..n_samples {
                let x_h = x.slice(s![h, .., ..]).to_owned();
                let x_t = x_h.t();
                let temp = w_i.t().dot(&x_t);
                let result = temp.dot(&w_k);
                tt[(h, f)] = result[(0, 0)];
            }

            let t = tt.slice(s![.., 0..f+1]).to_owned();

            let reg_param = self.a;
            let eye = Array2::eye(n_samples);
            let inv_term = t.dot(&t.t()) - reg_param * eye;
            let inv = inv_term.inv().map_err(|_| "Matrix inversion failed")?;

            let tt_dot_inv = t.t().dot(&inv);
            let y_col = y.clone().into_shape_with_order((n_samples, 1))
                .map_err(|_| "y reshape failed")?;
            let bf_scalar = tt_dot_inv.dot(&y_col);

            let n_components_used = f + 1;
            let mut bf = Array2::zeros((n_components_used, 1));
            let val = bf_scalar[(0, 0)];
            for i in 0..n_components_used {
                bf[(i, 0)] = val;
            }

            self.bf_array.push(bf.clone());

            let ww = w_k.dot(&w_i.t());

            for g in 0..n_samples {
                let mmas = tt[(g, f)] * &ww;
                let mut x_g = x.slice_mut(s![g, .., ..]);
                let x_g_old = x_g.to_owned();
                x_g.assign(&(x_g_old - mmas));
            }

            let t_bf = t.dot(&bf);
            for i in 0..n_samples {
                y[i] -= t_bf[(i, 0)];
                mass[i] += t_bf[(i, 0)];
            }
        }

        let diff = &mass - &y_copy;
        self.train_error = diff.mapv(|v: f64| v * v).sum() / n_samples as f64;
        Ok(())
    }

    pub fn predict(&self, xtest: &Array3<f64>) -> Result<Array1<f64>, &'static str> {
        let n_samples = xtest.shape()[0];
        let mut x = xtest.clone();
        let mut y = Array1::zeros(n_samples);
        let mut tt = Array2::zeros((n_samples, self.n_components));

        for f in 0..self.n_components {
            let w_k = &self.w_k[f];
            let w_i = &self.w_i[f];

            for h in 0..n_samples {
                let x_h = x.slice(s![h, .., ..]).to_owned();
                let x_t = x_h.t();
                let temp = w_i.t().dot(&x_t);
                let result = temp.dot(w_k);
                tt[(h, f)] = result[(0, 0)];
            }

            let t = tt.slice(s![.., 0..f+1]).to_owned();
            let ww = w_k.dot(&w_i.t());

            for g in 0..n_samples {
                let mmas = tt[(g, f)] * &ww;
                let mut x_g = x.slice_mut(s![g, .., ..]);
                let x_g_old = x_g.to_owned();
                x_g.assign(&(x_g_old - mmas));
            }

            let t_bf = t.dot(&self.bf_array[f]);
            for i in 0..n_samples {
                y[i] += t_bf[(i, 0)];
            }
        }

        Ok(y)
    }

    pub fn export_params(&self) -> NPLS1Params {
        NPLS1Params {
            n_components: self.n_components,
            a: self.a,
            bf_array: self.bf_array.clone(),
            train_error: self.train_error,
            w_k: self.w_k.clone(),
            w_i: self.w_i.clone(),
        }
    }

    pub fn get_w_k(&self) -> &[Array2<f64>] { &self.w_k }
    pub fn get_w_i(&self) -> &[Array2<f64>] { &self.w_i }
    pub fn get_bf_array(&self) -> &[Array2<f64>] { &self.bf_array }
    pub fn get_train_error(&self) -> f64 { self.train_error }
}

#[derive(Clone, Debug)]
pub struct NPLS1Params {
    pub n_components: usize,
    pub a: f64,
    pub bf_array: Vec<Array2<f64>>,
    pub train_error: f64,
    pub w_k: Vec<Array2<f64>>,
    pub w_i: Vec<Array2<f64>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_fit_predict() {
        let i = 5;
        let j = 2;
        let k = 3;
        let x = Array3::<f64>::ones((i, j, k));
        let y = Array1::<f64>::from_vec(vec![2.0; i]);
        let mut model = NPLS1::new(1, 3.0);
        let result = model.fit(&x, &y);
        assert!(result.is_ok());
        let pred = model.predict(&x).unwrap();
        assert_eq!(pred.len(), i);
        let params = model.export_params();
        assert_eq!(params.w_k.len(), 1);
        assert_eq!(params.w_i.len(), 1);
        assert_eq!(params.bf_array.len(), 1);
    }
}