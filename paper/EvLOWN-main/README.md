# EvLOWN: Weakly nonlinear oscillator Inference Framework

Codes and data for paper: Encoding cumulation to learn perturbative nonlinear oscillatory dynamics

Authors: Teng Ma, Ting-Ting Gao, Wei Cui*, Attilio Frangi, Gang Yan*, and Lin Zhao*

## Evolutionary learning oscillator with weak nonlinearity

This repository contains a slow-varying evolutions learning framework for inferring perturbative nonlinear oscillatory dynamics, and several numerical examples and real-world cases showcasing its usage.

## Numerical Example

Four numerical examples are invloved in Folder Example, including damped-harmonic oscillator, van der pol oscillator, duffing oscillator, weakly coupling oscillator.

## Vortex-induced Vibration

the EvLOWN approach is applied to observed VIV response data from a long-span suspension bridge, aiming to identify the governing equations that characterize the underlying VIV dynamics, including structural motion and vortex shedding interactions.

## Space station orbit dynamics

we infer the underlying orbital dynamics of two operational space stations, the China Tiangong Space Station (CSS) and the International Space Station (ISS), using observed trajectory data

## Getting start with your own data

```
# x [dim,N], data; t[N], time
import sys
sys.path.append("../Model")
import Model
model = Model.WeakNO(dim,library,library_name)
model.Get_frequency(x,t)
model.Get_Evolution(smooth_window = 1)
model.Library_rebuild()
model.optimize(sparse_threshold = 1e-2,
               stop_tolerance = 1e-3,
               step_tolerance=  1e-2,
               smooth_window=1)
print(model.Xi)
```
