# TrainingTemperatureExperiments
A repository to store experiments that explore how the "temperature" changes when a neural network groks. We explore simple vision and language models during supervised training and simple version of unsupervised training. When I say temperature, I mean, basically, the norm (or some metric of size) of the amount of "movement" in the network. If the network is changing a lot (which we measure by looking at gradient magnitude) then it is high "temperature" and it "cools down".

# Notes (MVP - tested for 20 epochs)
Because of log graph, everything which was seen, which was almost linear, was actually exponential (albeit with perhaps a small base).

- It seems like a handful of CNN layers cool down rather quickly, but after that there is a lot of noise; it is not easy to eyeball any sort of pattern after the initial cooling (which happens in really few steps, i.e. like order of 1 epoch).
- For some reason almost every batch norm goes back up; similarly, for the downsample, it goes back up as well.
- For some reason I get the vibe that biases are more noisy.
- Generally, later layers tend to take longer to come down (especially FC: FC is the slowest).
- Often, temperature drop dips are followed by rebounds... even when the later rebound is almost entirely flat

**Big question is why the dip, and will we see anything if we run for more epochs or try a different type of model or correlate with accuracy, sub-task accuracy, or anything else?** I am also curious to see which components correlate in their cooling and whether the "heat" moves around (i.e. if there are any dynamics of this quantity).

Look at/reproduce
- https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
- https://github.com/mechanistic-interpretability-grokking/progress-measures-paper/tree/main
- Look at how temperature moves around through the components and whether it is correlated with events