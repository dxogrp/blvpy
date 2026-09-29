# Example data sources

These small snapshots are committed so that the example notebooks execute
reproducibly and without network access. Hashes below are SHA-256 digests of
the downloaded source bytes and the derived CSV files.

## `stigler_diet.csv`

- **Upstream:** Google OR-Tools,
  [`stigler_diet.py`](https://github.com/google/or-tools/blob/100f66e6242ab8bf8d32feb8f3bf086db66ae2b5/ortools/linear_solver/samples/stigler_diet.py),
  commit `100f66e6242ab8bf8d32feb8f3bf086db66ae2b5`.
- **Retrieved:** 2026-09-29.
- **Upstream SHA-256:**
  `5642bb88e068a82168f170f98dfb39f7897f34faa49c26ce5afb61b5440d2cd1`.
- **Local SHA-256:**
  `5f02ba42b7ecacb1a4036ea71ef143e79f412446822b81d9528e42a2695dbe5c`.
- **Transformation:** the 77 food rows were converted from the Python data
  literal to CSV without changing values. Descriptive headers make explicit
  that the nine nutrient quantities are amounts obtained per 1939 dollar;
  calories are measured in thousands of kilocalories per dollar.
- **License and attribution:** Copyright 2010-2025 Google LLC. The upstream
  sample is distributed under the
  [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## `iris.csv`

- **Upstream:** UCI Machine Learning Repository,
  [`iris.zip`](https://archive.ics.uci.edu/static/public/53/iris.zip), using
  the corrected `bezdekIris.data` member rather than `iris.data`.
- **Retrieved:** 2026-09-29.
- **Archive SHA-256:**
  `d11fe30213d36434a0879aab7cb00ce3c812eb7ba2495874438abff7b7b762e9`.
- **Raw member SHA-256:**
  `0fed2a99db77ec533a62dc66894d3ec6df3b58b6a8f3cf4a6b47e4086b7f97dc`.
- **Local SHA-256:**
  `7866a81bd4601f74577a9cd0a7f03b35dcb0c361cf1e578a3930697524b0edeb`.
- **Transformation:** a descriptive header was added; all 150 measurements
  and the original `Iris-setosa`, `Iris-versicolor`, and `Iris-virginica`
  labels are unchanged. UCI identifies this as the corrected variant and
  notes errors in observations 35 and 38 of `iris.data`.
- **License and attribution:** the
  [UCI Iris dataset](https://doi.org/10.24432/C56C76) is licensed under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## `smard_germany_2019-06-21.csv`

- **Upstream:** the SMARD hourly chart-data endpoints with URL template
  `https://www.smard.de/app/chart_data/{filter}/DE/{filter}_DE_hour_1560722400000.json`.
  The selected filters are grid load (`410`), onshore wind generation
  (`4067`), offshore wind generation (`1225`), photovoltaic generation
  (`4068`), and their installed capacities (`186`, `4076`, and `188`).
- **Retrieved:** 2026-09-29. Every response reported metadata version 1 and
  contained 168 hourly values. Rows 96 through 119 were selected, covering
  `2019-06-21 00:00` through `23:00` CEST
  (`2019-06-20T22:00:00Z` through `2019-06-21T21:00:00Z`).
- **Endpoint metadata and SHA-256:**

  | filter | metadata `created` (ms) | raw SHA-256 |
  | ---: | ---: | --- |
  | 410 | 1741685783022 | `86003b362dfb756ec7318d35b1e2e72342c8f11c7056e65fdf4d16d51663f8c7` |
  | 4067 | 1742257686010 | `253df69aacb2dafd4a00a65a04fe548e300f6e76ae89bfbc67a737e73879bfc0` |
  | 1225 | 1670024765045 | `a8bba486d8a13ce457aac245e3566ff4a0c2fe97b0c0747f5eba2e37700cbd36` |
  | 4068 | 1742257687214 | `00a6dcc2e4fe7575431fa6d3b39eef83179db7583c20258c8903a331b69c591` |
  | 186 | 1717516762017 | `8f518ab2b17e41b95eae592dffa3f3767d01a55ab4d2b02aa087dd7773c8e07f` |
  | 4076 | 1651662040964 | `ec6c6889f111a97ddbf0e51b321aa85fa39d5214fe1f5cd5c542636c52521231` |
  | 188 | 1717516870720 | `457bb49166a07cc4812a2180bf539351aa2af9fc5ce02f745cd0ff0c9628848e` |

- **Local SHA-256:**
  `eb90472e9fbedbb69d5430288d86a43ba9433950aaa91792b904c704eee0da0a`.
- **Transformation:** the seven series were joined by timestamp, restricted
  to the stated local day, converted from millisecond timestamps to explicit
  UTC and CEST ISO-8601 strings, and given descriptive headers. No values were
  interpolated. The installed-capacity observations are constant during this
  day and are repeated in each row.
- **License and attribution:** “Bundesnetzagentur | SMARD.de”. SMARD market
  data may be shared and adapted under
  [CC BY 4.0](https://www.smard.de/en/datennutzung); the platform identifies
  ENTSO-E as its upstream data provider.
