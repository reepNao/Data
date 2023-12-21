"""
Probability:
------------
    - Probability is the measure of the likelihood that an event will occur in a Random Experiment.
    - Probability is quantified as a number between 0 and 1, where, loosely speaking, 0 indicates impossibility and 1 indicates certainty.
    - The higher the probability of an event, the more likely it is that the event will occur.
    - A simple example is the tossing of a fair (unbiased) coin. Since the coin is fair, the two outcomes ("heads" and "tails") are both equally probable; 
the probability of "heads" equals the probability of "tails"; and since no other outcomes are possible, the probability of either "heads" or "tails" is 1/2 (which could also be written as 0.5 or 50%).
    - These concepts have been given an axiomatic mathematical formalization in probability theory, which is used widely in such areas of study as mathematics, 
statistics, finance, gambling, science (in particular physics), artificial intelligence/machine learning, computer science, game theory, and philosophy to, for example, draw inferences about the expected frequency of events.
    - Probability theory is also used to describe the underlying mechanics and regularities of complex systems.


Statistics:
-----------
    - Statistics is a branch of mathematics dealing with data collection, organization, analysis, interpretation and presentation.
    - In applying statistics to a scientific, industrial, or social problem, it is conventional to begin with a statistical population or a statistical model to be studied.
    - Populations can be diverse groups of people or objects such as "all people living in a country" or "every atom composing a crystal".
    - Statistics deals with every aspect of data, including the planning of data collection in terms of the design of surveys and experiments.
    - See glossary of probability and statistics.
    - When census data cannot be collected, statisticians collect data by developing specific experiment designs and survey samples.
    - Representative sampling assures that inferences and conclusions can safely extend from the sample to the population as a whole.
    - An experimental study involves taking measurements of the system under study, manipulating the system, and then taking additional measurements using the same procedure to determine if the manipulation has modified the values of the measurements.
    - In contrast, an observational study does not involve experimental manipulation.

Generally sets are using for probability and statistics.

"""

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import scikit_learn as skl
import scikit_posthocs as sp


#operations on sets
#1 - Union
#2 - Intersection
#3 - Difference
#4 - Complement

my_Set0 = {13, 69, 81, 33, 42, 49, 44, 00, 98}
#1 - Union
my_Set1 = {13, 69, 81, 33, 42}
my_Set2 = {13, 49, 44, 33, 00, 98}
union_set1 = my_Set1.union(my_Set2)
union_set2 = my_Set2.union(my_Set1)

#2 - Intersection
my_Set3 = {13, 69, 81, 33, 42}
my_Set4 = {13, 49, 44, 33, 00, 98}
intersection_set1 = my_Set3.intersection(my_Set4)
intersection_set2 = my_Set4.intersection(my_Set3)

#3 - Difference
my_Set5 = {13, 69, 81, 33, 42}
my_Set6 = {13, 49, 44, 33, 00, 98}
difference_set1 = my_Set5.difference(my_Set6)
difference_set2 = my_Set6.difference(my_Set5)

#4 - Complement
my_Set7 = {13, 69, 81, 33, 42}
my_Set8 = {13, 49, 44, 33, 00, 98}
complement_set1 = my_Set0.difference(my_Set7)
complement_set2 = my_Set0.difference(my_Set8)

#disjoint sets - if two sets are not having any common elements then they are called as disjoint sets

#law mentioned by De Morgan: (A U B)' = A' intersection B'
#It means complement of union of two sets is equal to intersection of complement of two sets
universal_set = {1, 2, 3, 4, 5, 6, 7, 8, 9}
set_A = {1, 2, 3}
set_B = {7, 8, 9}
complement_of_union_ab = {4, 5, 6}
complement_of_union_ab == universal_set.difference(set_A.union(set_B))

set_A_complement = {4, 5, 6, 7, 8, 9}
set_B_complement = {1, 2, 3, 4, 5, 6}
intersection_of_ab_complement = {4, 5, 6}
intersection_of_ab_complement == set_A_complement.intersection(set_B_complement) 



"""
EXPERIMENT:
-----------
    1 - Random Experiment
    2 - Outcomes
    3 - Sample Space (all the outcomes of an experiment that is called sample space)
    {(H,T)}, {(1,2,3,4,5,6)}, {(H,1), (H,2), (H,3), (H,4), (H,5), (H,6),
                               (T,1), (T,2), (T,3), (T,4), (T,5), (T,6)} = ITs all Sample Space.
    4 - Event = can be any subset of sample space. except empty subset


PROBABILITY MODEL:
------------------
    1 - From Event to a Number
        Ω = {H,T}
        E1, E2 = {H}, {T}
        P(E1) = 0.5
        P(E2) = 0.5

        Ω2 = {1,2,3,4,5,6}
        P(e) = P(2) + P(4) + P(6)
    2 - Axioms of Probability
        -P(Ex) >= 0
        -P(E1 U E2) = P(E1) + P(E2) if E1 and E2 is disjoint. (E1.intersection(E2) = None)
            P(E1 U E2) = P(E1) + P(E2) - P(E1.intersection(E2))
        -P(Ω) = 1
        -P(A U B) <= P(A) + P(B)
    3 - Conditional Probability
        Ω = {1,2,3,4,5,6}
        Ei = {i}, 1<=i<=6
        P(i) = 1/6
        P(A\B) = P(A.intersectin(B)) / P(B)
    4 - Total Probability Theorem
        -B1, B2, B3 partitions of Ω
        -P(A) = P(A.intersection(B1)) + P(A.intersection(B2)) + P(A.intersection(B3))
              = P(A\B1)*P(B1) + P(A\B2)*P(B2) + P(A\B3)*P(B3)


BAYES RULE:
-----------
    1 - Bayes Rule
        P(A.intersection(B)) = P(B.intersection(A)) = P(A\B)*P(B) = P(B\A)*P(A)

        P(A\B) = P(B\A)*P(A) / P(B)
            -Almost all the classfiers, all the machine learning stands on this rule and that rules comes very naturally through the conditional probability


INDEPENDENCE:
-------------
    1 - Independence
    2 - Conditional Independence
"""
