# Feedback on the CART assignment

Any comments inserted directly into your files are marked `#CLAUDE>>` (written by
Claude, an AI) or `#DAN>>` (written by Dan). Find them all by searching for `>>`;
`grep -rn '>>' .` lists every one. They are ordinary code comments, so your code
runs exactly as it did before. Claude's comments carry no grade and Claude does
not grade; any grade for this assignment comes from Dan, at the end of his
section below.

## Claude Feedback

Your pruning section is the strongest part. The CV-error-vs-alpha plot has SE
bars, and the 1-SE pick takes the simplest tree under the line, which is the rule
done properly. Running both a manual CV loop and `cross_val_score` and checking
that they agree was a good instinct. Your comments also make it easy to follow
your thinking as you go.

Most worth working on:

- **Day 7: the test set.** The final forest is trained on `Xtest` and then
  scored on it. Fit on the training data, then predict the test set once. That
  single honest number is what the whole vault exists for.
- **Check your labels before trusting a perfect score.** The xgboost 0.0 comes
  from `ytrain=='M'`, a line carried over from the breast-cancer code, turning
  all your labels into 0. When a model looks too good to be true, look at what
  it was given.
- **Answer the Day-3 questions in comments**, especially why your full tree beat
  the default one on this dataset. Also save the simplified breast-cancer data at
  the end of `ExploreBreastCancerData.py`.

## Dan Feedback

Youve done everything, and most of it is right, but with a couple bugs that ended up 
having rather severe consequences. At once point your overwrote your response variable,
which I think is why you got a 0% out of sample accuracy. And then, on the test set,
you re-trained your RF model on the test set before evaluating it on the test. 

Please go back and look at the individual comments from Claude and me (search for ">>")
as you will learn something. Make sure you understand the sources of the bugs, so you
don't make the same/similar bug next time.

Grade: S