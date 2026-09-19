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

### Volume-profile screener criteria (frozen contract)

The foreign and broker volume-profile screeners accept these `screener_vprofile_criteria` query values:

* `vprofile_inside` — membership only: a recent close touched the selected profile zone.
* `vprofile_breakout` — exact `role=resistance`, `behavior=breakout_up`: approach from below, touch, then close above.
* `vprofile_breakdown` — exact `role=support`, `behavior=breakdown`: approach from above, touch, then close below.
* `vprofile_support_bounce` — exact `role=support`, `behavior=rejection`: approach from above, touch, then close above.
* `vprofile_resistance_rejection` — exact `role=resistance`, `behavior=rejection`: approach from below, touch, then close below.

The directional criteria require their exact role and behavior pair; a role alone,
membership alone, or a close still inside the zone is not selected.

Selected-zone annotations also include `vprofile_zone_strength` (absolute node flow
relative to the strongest node), `vprofile_zone_prominence` (raw peak/valley
prominence relative to the strongest absolute histogram flow),
`vprofile_zone_flow_share` (absolute node flow divided by total absolute profile
flow), and `vprofile_event_date` (latest close date as `YYYY-MM-DD`). These are
JSON-safe scalars; all four are `None` when no profile zone is selected. Existing
zone levels, role, behavior, distance, and touch count remain unchanged.

Copyright (c) 2023 Quantist.io. All rights reserved. This works, including all modifications made by any third party, is the property of Quantist.io and is protected by copyright law.
