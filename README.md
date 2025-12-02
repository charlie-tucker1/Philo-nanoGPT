# Philo-nanoGPT
I built this following along with Andrej Karpathy's lecture on building his nanoGPT. I trained this using the philosphy project database.

It's a simple Bigram Language Model that uses a implementation of multi head self-attention. I trained this on a t4 gpu for ~ 45 mins and got val
loss down to ~ 1.2 or so. It creates some semi-coherent text, I'm pretty happy with the results, I'll paste some of the generated text here:

    "The faculty of machine things which is immanent, but it follows itself as one should found form in these duty?"
    "The so a man represents of the substance of the consciousness, the world have strong produced manufacturer."
    "For yet our certain property, if it is a predicate share in which it is truly described by an unique dintifferent, then,
    the manufacturers of a school completely in different case, if we say, it however it is the other than composed supposed to observe itself."
    "(The question of the form of the first concept )(Even time), and still I shall find see on the constitutor of pretending them and social actions, at least a world."

As you can see, it doesn't really make any sense but, given the fact that its training data is incredibly old, I think it's pretty interesting.

