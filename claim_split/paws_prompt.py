PAWS_PROMPT = """
Please decompose the following sentence into decontextualized sentences while ensuring all information is retained and the wording is as unchanged as possible (please return the original sentence if it cannot be decomposed):

Original: When Phil Spector first listened to `` Stubborn Kind of Fellow '' , he was so excited that he lost control of his car while driving down the Sunset Boulevard with Jack Nitzsche .
Decomposed:
- When Phil Spector first listened to `` Stubborn Kind of Fellow '' , he was so excited that he lost control of his car.
- When Phil Spector first listened to `` Stubborn Kind of Fellow '' , he was driving down the Sunset Boulevard with Jack Nitzsche.

Original: Lloyd founded and conducted his business to begin selling toys and gifts , and he expanded the House of Lloyd , based in Grandview , Missouri , when the gift business grew .
Decomposed:
- Lloyd founded and conducted his business to begin selling toys and gifts.
- Lloyd expanded the House of Lloyd , based in Grandview , Missouri , when the gift business grew .

Original: The band was founded in Dunwoody , Georgia in 1999 , after the guitarist Cole Alexander and the bassist Jared Swilley left the Renegades , and guitarist Ben Eberbaugh left the Reruns .
Decomposed: 
- The band was founded in Dunwoody , Georgia in 1999.
- The band was founded after the guitarist Cole Alexander and the bassist Jared Swilley left the Renegades.
- The band was founded after the guitarist Ben Eberbaugh left the Reruns.

Original: {}
Decomposed:
"""
