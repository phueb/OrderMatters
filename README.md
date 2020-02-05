# Order-Matters

Research code to order documents by various corpus-statistics (e.g. lexical diversity, syntactic complexity).

Many practitioners ignore the order in which their learning models are trained on data by shuffling the input prior to training.
Yet, in many cases, learning may benefit from ordered training data.
This is due to the progressive differentiation underlying the learning dynamics of many neural networks.
 
 ## Idea
 
 1. Define subcategory structure
 2. Put those examples first which provide least superordinate category information (large conditional entropy)
 
TODO make markdown figure here showing three boxes diagram [context]<-CE-[target]<-CE-[net-word]
