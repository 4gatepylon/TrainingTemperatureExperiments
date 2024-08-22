# TrainingTemperatureExperiments
A repository to store experiments that explore how the "temperature" changes when a neural network groks. We explore simple vision and language models during supervised training and simple version of unsupervised training. When I say temperature, I mean, basically, the norm (or some metric of size) of the amount of "movement" in the network. If the network is changing a lot (which we measure by looking at gradient magnitude) then it is high "temperature" and it "cools down".

# Usage
1. Install dependencies: `python3 -m venv .venv && source .venv/bin/activate && pip3 install -r requirements.txt`
2. `resnet/` code is self-explanatory
3. `induction_heads/` is run as a Jupyter notebook in VSCode using the Jupyter plugin which lets you define cell start lines using `# %%` in python (this one can take up a lot of disk space: read the comments)
4. Grokking `project-measures-...` is not used right now

# Notes 
(This time around was an MVP - I just tested for 20 epochs)

Because of log graph, everything which was seen, which was almost linear, was actually exponential (albeit with perhaps a small base).

- It seems like a handful of CNN layers cool down rather quickly, but after that there is a lot of noise; it is not easy to eyeball any sort of pattern after the initial cooling (which happens in really few steps, i.e. like order of 1 epoch).
- For some reason almost every batch norm goes back up; similarly, for the downsample, it goes back up as well.
- For some reason I get the vibe that biases are more noisy.
- Generally, later layers tend to take longer to come down (especially FC: FC is the slowest).
- Often, temperature drop dips are followed by rebounds... even when the later rebound is almost entirely flat

**Big question is why the dip, and will we see anything if we run for more epochs or try a different type of model or correlate with accuracy, sub-task accuracy, or anything else?** I am also curious to see which components correlate in their cooling and whether the "heat" moves around (i.e. if there are any dynamics of this quantity).

### Random Questions For Future Exploration Out of Scope
- Do gradients more or less align with eachother over training? i.e. is this training in roughly a straight line or not? Might there be some shared center or curvature? etc...
- There is a shit-load of related work cited in the https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html paper pertaining to the mathematics of the loss landscape and training dynamics of deep neural networks; this could be valuable to learn/understand for this project, but it's not of the highest priority for the initial experiment
- Would it be equivalent or viable, in some capacity, to train the early layers first (with a smaller models), then the later layers, and so on?
- Do the weights changes exhibit variance from the average change?
- If we do a fourier analysis of gradient updates per-weight, could we get any interesting results? For example, could we discover any sorts of orbiting behavior?

Look at/reproduce
- https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html (mostly done, but need training)
- https://github.com/mechanistic-interpretability-grokking/progress-measures-paper/tree/main
- Look at how temperature moves around through the components and whether it is correlated with events