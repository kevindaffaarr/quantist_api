## Backend of Quantist.io: Capturing The Silhouette of Data
_Democratize the information of Indonesia Stocks and more to investors from retail to institution._

Consists of high-end analysis tools based on data with top-down analysis:
* Conservation of money (forex/crypto - bond - stocks)
* External factor (commodities, national/international issues, sentiments)
* Fundamental: business performance, corporate action
* Transaction: foreign flow, broker summary clustering, done trade, 5% shares holder, share holder distribution
* Behaviour: supply & demand, momentum, trend, time

For more information, you can find us at:
* [Website](https://quantist.io)
* [Email](mailto:kevindaffaar@quantist.io)

### Volume-profile screener criteria

The foreign and broker volume-profile screeners accept these `screener_vprofile_criteria` query values:

* `vprofile_inside` — recent closes touched the selected profile zone.
* `vprofile_breakout` — price approached resistance, touched it, and closed above it.
* `vprofile_breakdown` — price approached support, touched it, and closed below it.
* `vprofile_support_bounce` — price touched support and closed back above it (`role=support`, `behavior=rejection`).
* `vprofile_resistance_rejection` — price touched resistance and closed back below it (`role=resistance`, `behavior=rejection`).

The bounce and rejection criteria require both the role and rejection behavior; a role alone or a close still inside the zone is not selected.

Copyright (c) 2023 Quantist.io. All rights reserved. This works, including all modifications made by any third party, is the property of Quantist.io and is protected by copyright law.
