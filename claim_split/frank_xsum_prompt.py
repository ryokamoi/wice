FRANK_XSUM_PROMPT = """
Please decompose the following sentence into decontextualized sentences while ensuring all information is retained and the wording is as unchanged as possible (please return the original sentence if it cannot be decomposed):

Original: the confederation of african football ( caf ) and morocco have returned to the competition after a row in the confederation of african football broke down.
Decomposed: 
- the confederation of african football (caf) has returned to the competition after a row in the confederation of african football broke down.
- morocco has returned to the competition after a row in the confederation of african football broke down.

Original: a fracking operation in lancashire has been suspended after a gas leak was found at the site.
Decomposed: 
- a fracking operation in lancashire has been suspended after a gas leak was found at the site.

Original: southampton city council has been told to pay hundreds of millions of pounds a day of applying for a licence because of a lack of funding.
Decomposed: 
- southampton city council has been told to pay hundreds of millions of pounds a day of applying for a licence.
- this payment from southampton city council is a result of a lack of funding.

Original: a driver who knocked down a lollipop lady broke down in tears in court as he said he was ``truly sorry\\'\\'for her and her family.
Decomposed: 
- a driver who knocked down a lollipop lady broke down in tears in court.
- a driver who knocked down a lollipop lady said he was ``truly sorry\\'\\'for her and her family.

Original: {}
Decomposed:
"""
