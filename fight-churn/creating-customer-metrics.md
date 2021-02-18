# Customer metrics

We want to move from event data to metrics summarising user behaviour.

## On metrics

__Summary Sean Taylor's post on "designing and evaluating metrics"__ [ref](https://medium.com/@seanjtaylor/designing-and-evaluating-metrics-5902ad6873bf)

> A metric is simultaneously 1) a designed artifact, 2) a lens through which we observe phenomena, and 3) way we set and monitor goals. 
> 
> Investments in our ability to capture data and measure outcomes often precede step-function changes in our understanding of the world and the ability to better solve problems.

Five key properties of metrics:

__(1) Cost__ can entail money, calendar time, employee time, user time (interrupting users to ask them stuff), computation, or technical debt.

- In many cases we can trade time, money, or effort for better measurement. This tradeoff is challenging to manage because we must also estimate the payoff from having better metrics, and how that may propagate into downstream product or decision quality

__(2) Simplicity__, metrics are designed artifacts. The worst possible metric is one that people mistrust, second-guess, or ignore. 

- Metrics can often be improved through normalization (which tends to focus them). However, finding appropriate denominators can be extremely challenging
- Metrics can be made worse, through combination. Metrics should represent one thing (though may encapsulate and represent many things)

__(3) Faithfulness__, there are an unfortunately large number of ways your measurement can fail to accurately represent the concept you care about.

- Two of the most important ones I have observed in practice are metrics without construct validity or that have some kind of sampling bias
  - Metrics without construct validity measure the wrong thing
  - Measures with sampling bias measure it for the wrong set of units (e.g. people, items, events, etc)

For example, there is evidence that clicks on display ads are not predictive of sales. If you used clicks as a metric for your ad campaign, you would optimize for an irrelevant outcome; clickers do not resemble buyers.

__(4) Precision__, more precision is better and noisy metrics mean we can’t separate the signal from the noise.

Three things are useful to know about precision:

- You can gain substantial precision through transformations of metrics, either through taking logs, winsorizing or "variance-stabilizing transformations"
- Normalizations can substantially improve precision of a metric. If the numerator is very skewed and the denominator is as well, then the ratio creates a much lower variance metric
- Summing or average several metrics is useful for precision. If you have a few relatively uncorrelated ways of measuring the same thing, their sum will be less noisy. The price is reduced simplicity, and perhaps less causal proximity

> Often there’s an inherent tradeoff between precision and faithfulness. Although they are what we ultimately care about, metrics that capture financial outcomes (sales, revenue, or profit) can be quite noisy because of skewed distributions. Counting discrete outcomes like transactions or unique customers (which are binarizing a continuous outcome) will have bounded variance.

__(5) Causal Proximity__, a good metric can be affected by causes under your control.

> I use “proximity” to capture the idea that a metric “close” in “causal space” (e.g. in a path along a causal DAG) to policies you are capable of changing.

We must choose a metric with higher proximity and rely on a theory about how that is useful for some ultimate goal — a sacrifice of faithfulness. We sometimes call this strategy a proxy metric, acknowledging that it may not be exactly what we care about, but that is a concept on which we can detect effects.

__"Good behavioural metrics are the most important step in a successful churn analysis"__ _-- Carl Gold_

## Creating customer metrics

Feature engineering - find some notes

Use sliding windows, for example, rather than a monthly calculation use a moving 28 day window.

Human behaviour often follows weekly cycles. Consequently, it is best to measure using time windows that are multiples of 7 days.

## QA

It's important to run quality assurance (QA) tests on the underlying events data. Customer event data in data warehouses is often unreliable. This unreliability can manifest in different ways; for example, events can be lost on the network before they reach the data warehouse. In general, event data does not receive a lot of scrutiny.