//! Einstein summation convention.

use std::collections::HashMap;

use crate::Tensor;
use crate::reduce::AxisSpec;

type Result<T> = crate::Result<T>;

fn argsort<T: Ord>(slice: &[T]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..slice.len()).collect();
    indices.sort_by(|&a, &b| slice[a].cmp(&slice[b]));
    indices
}

impl Tensor {
    pub fn einsum(formula: &str, operands: &[&Tensor]) -> Result<Tensor> {
        let mut xs: Vec<Tensor> = operands.iter().map(|t| (*t).clone()).collect();
        let formula = formula.replace(' ', "");

        // Expand ellipsis
        let formula = if formula.contains("...") {
            let all_chars: std::collections::HashSet<char> =
                formula.chars().filter(|c| c.is_ascii_alphabetic()).collect();
            let ell: String = ('a'..='z').chain('A'..='Z').filter(|c| !all_chars.contains(c)).collect();

            let lhs = formula.split("->").next().unwrap();
            let input_strs: Vec<&str> = lhs.split(',').collect();

            let ell_n: Vec<usize> = input_strs
                .iter()
                .zip(xs.iter())
                .map(|(s, x)| {
                    if s.contains("...") {
                        let ndim = x.ndim().unwrap();
                        let non_ell_chars = s.len() - 3; // subtract "..."
                        ndim.saturating_sub(non_ell_chars)
                    } else {
                        0
                    }
                })
                .collect();

            let max_ell_n = *ell_n.iter().max().unwrap_or(&0);

            let mut new_inputs: Vec<String> = Vec::new();
            for (i, s) in input_strs.iter().enumerate() {
                let replacement = &ell[max_ell_n - ell_n[i]..max_ell_n];
                new_inputs.push(s.replace("...", replacement));
            }

            let new_lhs = new_inputs.join(",");

            // Build auto output: sorted chars that appear exactly once in lhs and are not ellipsis chars
            let ell_chars: std::collections::HashSet<char> = ell[..max_ell_n].chars().collect();
            let auto: String = {
                let mut chars: Vec<char> = lhs
                    .chars()
                    .filter(|c| {
                        c.is_ascii_alphabetic() && *c != '.' && lhs.matches(*c).count() == 1 && !ell_chars.contains(c)
                    })
                    .collect();
                chars.sort();
                chars.into_iter().collect()
            };

            if formula.contains("->") {
                let rhs = formula.split("->").nth(1).unwrap();
                let new_rhs = rhs.replace("...", &ell[..max_ell_n]);
                format!("{new_lhs}->{new_rhs}")
            } else {
                format!("{new_lhs}->{}{auto}", &ell[..max_ell_n])
            }
        } else {
            formula
        };

        // Split into lhs and rhs
        let (lhs, rhs) = if formula.contains("->") {
            let parts: Vec<&str> = formula.split("->").collect();
            (parts[0].to_string(), parts[1].to_string())
        } else {
            let auto: String = {
                let mut chars: Vec<char> =
                    formula.chars().filter(|c| c.is_ascii_alphabetic() && formula.matches(*c).count() == 1).collect();
                chars.sort();
                chars.into_iter().collect()
            };
            (formula.clone(), auto)
        };

        let mut inputs: Vec<String> = lhs.split(',').map(|s| s.to_string()).collect();

        // Trace: diagonal for repeated letters
        for i in 0..inputs.len() {
            let mut s = inputs[i].clone();
            let mut x = xs[i].clone();
            let unique_chars: Vec<char> = {
                let mut seen = std::collections::HashSet::new();
                s.chars().filter(move |c| seen.insert(*c)).collect()
            };
            for c in unique_chars {
                while s.matches(c).count() > 1 {
                    let j = s.find(c).unwrap();
                    let k = s[j + 1..].find(c).unwrap() + j + 1;
                    let shape = x.shape()?;
                    let n = shape[j].as_const().unwrap();
                    let ndim = x.ndim()?;

                    if ndim > 2 {
                        // permute so j,k are last two dims
                        let mut perm: Vec<isize> =
                            (0..ndim).filter(|&d| d != j && d != k).map(|d| d as isize).collect();
                        perm.push(j as isize);
                        perm.push(k as isize);
                        x = x.try_permute(&perm)?;

                        // flatten last two dims
                        x = x.flatten_range(-2, -1)?;

                        // pad with n zeros at end of last dim
                        let new_ndim = x.ndim()?;
                        let mut padding = vec![(0isize, 0isize); new_ndim];
                        padding[new_ndim - 1] = (0, n as isize);
                        x = x.try_pad(&padding)?;

                        // unflatten last dim into [n, n+1]
                        x = x.unflatten(-1, &[n as isize, (n + 1) as isize])?;

                        // take element 0 along last dim (shrink + squeeze)
                        let cur_ndim = x.ndim()?;
                        let mut ranges: Vec<(isize, isize)> =
                            x.shape()?.iter().map(|d| (0, d.as_const().unwrap() as isize)).collect();
                        ranges[cur_ndim - 1] = (0, 1);
                        x = x.try_shrink(&ranges)?;
                        x = x.try_squeeze(Some(-1))?;
                    } else {
                        // 2D diagonal: use flatten + stride approach
                        // For a [n, n] matrix, diagonal = flatten then take every (n+1)th element
                        x = x.flatten()?;
                        let stride = n + 1;
                        x = x.try_stride(&[stride as isize])?;
                    }

                    // Remove the second occurrence of c from s
                    s = format!("{}{}", &s[..k], &s[k + 1..]);
                }
            }
            inputs[i] = s;
            xs[i] = x;
        }

        // Build size map
        let mut sz: HashMap<char, usize> = HashMap::new();
        for (s, x) in inputs.iter().zip(xs.iter()) {
            let shape = x.shape()?;
            for (c, dim) in s.chars().zip(shape.iter()) {
                let dim_val = dim.as_const().unwrap();
                sz.insert(c, dim_val);
            }
        }

        let mut alpha: Vec<char> = sz.keys().copied().collect();
        alpha.sort();

        // Align, multiply, sum, permute
        let full_shape: Vec<isize> = alpha.iter().map(|c| sz[c] as isize).collect();

        let mut aligned: Vec<Tensor> = Vec::new();
        for (s, x) in inputs.iter().zip(xs.iter()) {
            if s.is_empty() {
                aligned.push(x.clone());
            } else {
                let mut sorted_chars: Vec<char> = s.chars().collect();

                let mut char_positions: Vec<(char, usize)> = s.chars().enumerate().map(|(i, c)| (c, i)).collect();
                char_positions.sort_by_key(|(c, _)| *c);
                let perm: Vec<isize> = char_positions.iter().map(|(_, pos)| *pos as isize).collect();

                sorted_chars.sort();

                let x = x.try_permute(&perm)?;

                // Reshape: insert 1s for missing dims
                let reshape: Vec<isize> =
                    alpha.iter().map(|c| if sorted_chars.contains(c) { sz[c] as isize } else { 1 }).collect();
                let x = x.try_reshape(&reshape)?;

                // Expand to full shape
                let x = x.try_expand(&full_shape)?;
                aligned.push(x);
            }
        }

        // Multiply all aligned tensors
        let mut product = aligned[0].clone();
        for t in aligned.iter().skip(1) {
            product = product.try_mul(t)?;
        }

        // Sum over axes not in rhs
        let sum_axes: Vec<isize> =
            alpha.iter().enumerate().filter(|(_, c)| !rhs.contains(**c)).map(|(i, _)| i as isize).collect();

        if !sum_axes.is_empty() {
            product = product.sum_with().axes(AxisSpec::Multiple(sum_axes)).call()?;
        }

        // Permute to match rhs order
        if !rhs.is_empty() {
            let rhs_chars: Vec<char> = rhs.chars().collect();
            let rhs_order = argsort(&argsort(&rhs_chars));
            let perm: Vec<isize> = rhs_order.iter().map(|&i| i as isize).collect();
            product = product.try_permute(&perm)?;
        }

        Ok(product)
    }

    /// Flatten a range of dimensions (inclusive).
    fn flatten_range(&self, start: isize, end: isize) -> Result<Tensor> {
        let shape = self.shape()?;
        let ndim = shape.len();
        let start = Self::normalize_axis(start, ndim)?;
        let end = Self::normalize_axis(end, ndim)?;

        let mut new_shape: Vec<isize> = Vec::new();
        let mut merged = 1isize;
        for (i, d) in shape.iter().enumerate() {
            let v = d.as_const().unwrap() as isize;
            if i >= start && i <= end {
                merged *= v;
                if i == end {
                    new_shape.push(merged);
                }
            } else {
                new_shape.push(v);
            }
        }
        self.try_reshape(&new_shape)
    }

    /// Stride along each dimension (take every nth element).
    fn try_stride(&self, strides: &[isize]) -> Result<Tensor> {
        let shape = self.shape()?;
        let ndim = shape.len();
        assert_eq!(strides.len(), ndim);

        let mut result = self.clone();
        for (dim, &stride) in strides.iter().enumerate() {
            if stride == 1 {
                continue;
            }
            let cur_shape = result.shape()?;
            let dim_size = cur_shape[dim].as_const().unwrap();
            let new_dim_size = dim_size.div_ceil(stride as usize);

            // Reshape dim into [new_dim_size, stride], take [:, 0]
            let mut new_shape: Vec<isize> = cur_shape.iter().map(|d| d.as_const().unwrap() as isize).collect();

            // Pad if needed so dim is evenly divisible by stride
            let padded_size = new_dim_size * stride as usize;
            if padded_size != dim_size {
                let mut padding = vec![(0isize, 0isize); result.ndim()?];
                padding[dim] = (0, (padded_size - dim_size) as isize);
                result = result.try_pad(&padding)?;
                new_shape[dim] = padded_size as isize;
            }

            // Unflatten dim into [new_dim_size, stride]
            new_shape.splice(dim..=dim, [new_dim_size as isize, stride]);
            result = result.try_reshape(&new_shape)?;

            let mut ranges: Vec<(isize, isize)> =
                result.shape()?.iter().map(|d| (0, d.as_const().unwrap() as isize)).collect();
            ranges[dim + 1] = (0, 1);
            result = result.try_shrink(&ranges)?;
            result = result.try_squeeze(Some((dim + 1) as isize))?;
        }
        Ok(result)
    }
}
