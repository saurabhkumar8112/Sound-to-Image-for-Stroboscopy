# Sound-to-Image-for-Stroboscopy
Standard diagnosis process of vocal defect involves inserting expensive specialzed cameras deep down the vocal pipe to take images of a person speaking then a professional(a doctor) does the diagnosis to check whether the person got any deformalities other than a normal vocal image. As it seems the whole process is highlt expensive, invasive and also requires a lot of expertise. It is also infeasible in the case of infact since you cannot simply insert a tune down their gut. 

We explore another possibility through the idea of latent relationships in hidden spaces. We belive that different people which have voice deformalities will have different speech pattern than those of normal people. Let us define the space of voice signal be z, and the space of this observed images as X. So their should exist a one-one function g, which takes z to X as X=g(z). We try to find this function g with the help of a GAN(generative adversarial network) setting. GANs have shown tremendous capabilities when trying to find a generator function satisfying properties that we want.

So the plan is to map sound pattern of each patient then pass it onto a function "g" learned by the neural network and get an image corresponding to that X, then a professional can simply diagnose them without the use of any expensive camera or invasive test.
 # z-g(GAN)-X-diagnosis
