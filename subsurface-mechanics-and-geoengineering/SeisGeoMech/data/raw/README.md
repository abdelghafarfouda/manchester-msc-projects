# data/raw

## 48_10b-_9_jwl_JWL_FILE_1682139.las

The supplied well log, copied **unmodified** from the `Seismic` teaching folder.

UK well 48/10b-9, southern North Sea. Operator BP, spudded 1990-08-26, LAS 2.0,
18,975 depth samples from 181.10 to 12,631.23 ft, NULL = -999.25.

Curves present: `DEPT, CALI, ILM, ILD, DT, TENS, CGR, DTL, RHOB, DRHO, PEF, NPHI,
THOR, URAN, GR, POTA`.

Two facts about this file decide the shape of the project:

* `RHOB` is logged over only 3,614.2-3,850.0 m (about 6 % of the well), while `DT`
  covers 766-3,835 m. Their overlap - about 221 m - is the only place Gardner's
  relation can be tested against a measurement.
* There is **no shear sonic** (`DTS`). Without it, `K` and `G` cannot be separated,
  so no Young's modulus, Poisson's ratio or horizontal stress is computed for this
  well anywhere in the project.

No other dataset is used. The teaching folders themselves are not part of this
repository.
