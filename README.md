# TrainingTemperatureExperiments
A repository to store experiments that explore how the "temperature" changes when a neural network groks. We explore simple vision and language models during supervised training and simple version of unsupervised training. When I say temperature, I mean, basically, the norm (or some metric of size) of the amount of "movement" in the network. If the network is changing a lot (which we measure by looking at gradient magnitude) then it is high "temperature" and it "cools down".

# Notes 2024-08-15 
(This time around was an MVP - I just tested for 20 epochs)

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

# Notes 2024-08-20
## Agenda
### High Level
The agenda is to basically make a very small experiment that I can publish on LW as a colab notebook+github as a sort of practice round for doing small, iterative, Mech. Interp. experiments (more of which I intend to do later). The purpose of this experiment is not so much to provide utility, but more instead to answer a sort of high-level question about DL science: are there any relations between the timing and amount of change in a neural network's components' weight per-component? Is there any way in which we can think of this as some sort of dynamical system where different components interact with each other in a way that may be informative?

One possible use of this sort of exploration is that we could try to understand grokking but at the end of the day the real goal is just to understand if the idea of tracking component-to-component training dynamics in terms of "movement" is even a useful lens to look through this at all.

### Scope
More specific questions here and experimental design.
- Reproduce in context learning and induction heads for small networks; look for grokking and try to see if we correlate with temperature drops
    1. Confirm the superiority of 2+ layer neural networks after training using transformer lens tutorial on those networks
    2. Randomize and retrain; Be able to train a 1-3 layer set of networks and confirm that you see the phase change


After that TBD but basically small algorithmic tasks and visual tasks with known things to grok and then measuring the grok


### Random Questions For Future Exploration Out of Scope
- Do gradients more or less align with eachother over training? i.e. is this training in roughly a straight line or not? Might there be some shared center or curvature? etc...
- There is a shit-load of related work cited in the https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html paper pertaining to the mathematics of the loss landscape and training dynamics of deep neural networks; this could be valuable to learn/understand for this project, but it's not of the highest priority for the initial experiment