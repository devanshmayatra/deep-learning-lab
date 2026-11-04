

fn logsumexp(x: &[f64]) -> f64 {
    
    if x.is_empty(){
        return f64::NEG_INFINITY;
    }
    let c = x.iter().copied.fold(f64::NEG_INFINITY, f64::max);

    if c.is_infinite() && c.is_sign_negative(){
        return f64::NEG_INFINITY;
    }

    let sum_exp: f64 = x.iter().map(|&xi| (xi-c).exp()).sum();
    c + sum_exp.ln()

}

// --- Test: Naive vs Stable ---
let logits = vec![1000.0, 1001.0, 1002.0];

// Naive method: exp(1002.0) overflows f64
let naive: f64 = logits.iter().map(|x| x.exp()).sum::<f64>().ln();

// Stable LogSumExp
let stable = logsumexp(&logits);

println!("Naive:  {}", naive);  // Output: inf
println!("Stable: {}", stable); // Output: 1002.4076059644443