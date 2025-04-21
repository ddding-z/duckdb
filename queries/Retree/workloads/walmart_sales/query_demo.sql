SELECT count(*)
FROM weekly_logs w JOIN stores s ON w.store = s.store
JOIN features f ON w.store = f.store AND w.date = f.date
WHERE predict('weekly_sales.onnx', data) > ?
GROUP BY w.store, w.dept;
