# Results

* Do not use smoothing-based solvers if you only care about the terminal value. Use filtering-based solvers instead.
* Dynamic calibration seems to make the solvers perform worse. Use the non-dynamic versions if possible.
* On non-stiff problems, use low-order EK0 versions for low precision, and high-order EK1 versions for high precision. Less than 5 derivatives count as "low order", and more than 5 derivatives count as "high order".
* TBC.
