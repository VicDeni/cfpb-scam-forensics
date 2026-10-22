-- Queries for a Databricks AI/BI dashboard (Dashboards → Create dashboard → Data → add each
-- query as a dataset, then add a visualization for it). Change workspace.cfpb if you used a
-- different catalog or schema.

-- 1. Complaints per month by product  → line chart (x: month, y: n_complaints, color: product_label)
SELECT month, product_label, n_complaints
FROM workspace.cfpb.gold_complaint_trends
WHERE product_label <> 'EXCLUDED'
ORDER BY month;

-- 2. On-time response rate by product, last 12 months  → bar chart
SELECT product_label, ROUND(100 * SUM(pct_timely * n_complaints) / SUM(n_complaints), 1) AS pct_timely
FROM workspace.cfpb.gold_complaint_trends
WHERE month >= add_months((SELECT MAX(month) FROM workspace.cfpb.gold_complaint_trends), -11)
  AND product_label <> 'EXCLUDED'
GROUP BY product_label
ORDER BY pct_timely;

-- 3. Most-complained-about companies per product  → table (add a product_label filter)
SELECT product_label, rank, company, n_complaints
FROM workspace.cfpb.gold_top_companies
ORDER BY product_label, rank;

-- 4. Routing accuracy and cost by approach  → bar chart + table (the headline result)
SELECT approach, prompt_version, n,
       ROUND(100 * accuracy, 1) AS accuracy_pct,
       ROUND(100 * macro_f1, 1) AS macro_f1_pct,
       ROUND(100 * parse_error_rate, 1) AS parse_error_pct,
       ROUND(cost_per_1k_usd, 3) AS cost_per_1k_usd
FROM workspace.cfpb.gold_model_comparison
ORDER BY macro_f1 DESC;

-- 5. F1 by product class for each approach  → grouped bar chart (x: label, y: f1, color: approach)
SELECT approach, label, ROUND(f1, 3) AS f1, support
FROM workspace.cfpb.gold_per_class_metrics
ORDER BY label, approach;

-- 6. Most common mistakes  → table
SELECT approach, true_label, predicted, `count`
FROM workspace.cfpb.gold_confusions
ORDER BY `count` DESC
LIMIT 20;

-- 7. Latest data-quality run  → table (counter: number of failed checks)
SELECT check_name, severity, passed, observed, threshold
FROM workspace.cfpb.dq_results
WHERE run_at = (SELECT MAX(run_at) FROM workspace.cfpb.dq_results)
ORDER BY passed, severity;

-- ===================== Part 2: payment-scam forensics =====================

-- 8. What the keyword/issue matches really are, by month  → stacked bar (x: month, color: fraud_type)
SELECT month, fraud_type, n_complaints
FROM workspace.cfpb.gold_scam_mix_by_month
ORDER BY month;

-- 9. Scam stories over time  → line chart (x: month, y: n_complaints, color: scam_type)
SELECT month, scam_type, n_complaints
FROM workspace.cfpb.gold_scam_types_by_month
WHERE scam_type NOT IN ('PARSE_ERROR', 'not_applicable')
ORDER BY month;

-- 10. Reported losses by payment method  → bar chart of median_loss_usd + table
SELECT fraud_type, payment_method, n_complaints, n_with_amount,
       ROUND(median_loss_usd) AS median_loss_usd, ROUND(p90_loss_usd) AS p90_loss_usd
FROM workspace.cfpb.gold_scam_losses_by_method
WHERE n_with_amount >= 10
ORDER BY fraud_type, median_loss_usd DESC;

-- 11. Refund denial rate: tricked vs. unauthorized, and by scam story  → bar chart
SELECT fraud_type, scam_type, n_mentioning_refund,
       ROUND(100 * pct_denied, 1) AS pct_denied, ROUND(100 * pct_refunded, 1) AS pct_refunded
FROM workspace.cfpb.gold_scam_refunds
WHERE n_mentioning_refund >= 10
ORDER BY pct_denied DESC;

-- 12. Did complaining help? Monetary relief by refund status before the complaint  → bar chart
SELECT fraud_type, refund_status, n_complaints, ROUND(100 * pct_monetary_relief, 1) AS pct_monetary_relief
FROM workspace.cfpb.gold_scam_relief_after_complaint
WHERE refund_status NOT IN ('PARSE_ERROR', 'not_applicable')
ORDER BY fraud_type, refund_status;

-- 13. How much to trust the numbers: extraction accuracy vs. hand labels  → table
SELECT prompt_version, field, n, ROUND(100 * accuracy, 1) AS accuracy_pct,
       CONCAT(ROUND(100 * ci_low), '-', ROUND(100 * ci_high), '%') AS ci_95
FROM workspace.cfpb.gold_scam_extraction_accuracy
ORDER BY prompt_version, field;
