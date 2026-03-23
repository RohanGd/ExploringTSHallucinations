Hallucinations in time series as defined by https://openreview.net/forum?id=vgfG8sEVf9 are forecasts whose dynamics do not conform to the dynamics of the context.

This repo is replicating the work proposed in the above paper to see if the claims hold as they do not provide source code.

### Project Structure:

#### datasets
##### [synthetic.py](./datasets/synthetic.py)
Creates common signal patterns - sinusoidal, square, sawtooth etc, ..
##### [m4.py](./datasets/m4.py)
Loads the datasets from the m4 competition.

For now context and forecast horizon is set to 500 and 64 as described in the Appendix
def __init__(
        self,
        context_len: int = 500,
        forecast_horizon: int = 64,
        seed: int = 42,
    ):

#### dataloaders
simply returns a pytorch dataloader

#### models
defines instances of chronos or timesfm
### TODO: to create activation heatmaps we will have to put chronos/timesfm source code here. RIght nw it is using the hugging face transformers library.(still local).

#### pipeline
basically can run dataset creation, initialize dataloaders and model and then save forecasts
see [file](./pipeline/forecast_pipeline.py) to see how to use

#### evaluation
evaluate_from_csv.py

have defined some metrics here. Mostly claude.ai created metrics

#### [knowledge rules](./knowledge_rules.py)
This file is used to detect if forecast/ target is hallucinating w.r.t. context. THis implemetns the knowlledge set as given by the paper. THe thresholds are not revealed in the paper and are manually set to 0.5 for now.

#### visualize.py 
This should create plots for context vs forecast vs target to visually inspect.

## TODO:
Evaluate visually if this even makes any sense. Try changing thresholds in knowledge set. Changing context length and forecast horizons.
discuss why this is not followed in the paper:
M4 Competition Forecast Horizons (https://github.com/GregorioMendozaSerrano/M4-Competition-Time-Series-Forecasting)
The required forecast horizon for each frequency is as follows:
Yearly: 6 years
Daily: 14 days
Hourly: 48 hours
Weekly: 13 weeks
Quarterly: 8 quarters 

Minimum Context Lengths (History)
The competition provided training sets with the following minimum observation counts to ensure models had enough context to generate valid forecasts: 
Yearly: 13 observations
Daily: 93 observations
Hourly: 700 observations
Weekly: 80 observations
Quarterly: 16 observations