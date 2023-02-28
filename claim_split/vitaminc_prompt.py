VITAMINC_PROMPT = f"""
Please decompose the following sentence into decontextualized sentences while ensuring all information is retained (please return the original sentence if it cannot be decomposed):

Original: Focus ( film ) , earned $ 6.4 million on its first day , $ 7.6 million the following day and $ 4.6 million on its third day , for a total of $ 18.7 million on its first weekend in 3,323 theaters and finishing on top at the box office .
Decomposed:
- Focus ( film ) earned $ 6.4 million on its first day .
- Focus ( film ) earned $7.6 million the following day.
- Focus ( film ) earned $4.6 million on its third day.
- Focus ( film ) earned a total of $18.7 million on its first weekend.
- Focus ( film ) was shown in 3,323 theaters.
- Focus ( film ) was finishing on the top film at the box office.

Original: Based on less than 30 ; 7 and 12 reviews respectively , the Microsoft windows version scored above 83 % ; the PlayStation 4 version scored below 81 % ; and the Xbox One version scored below 77.5 % .
Decomposed:
- Based on less than 30 reviews, the Microsoft windows version scored above 83 %.
- Based on less than 7 reviews, the PlayStation 4 version scored below 81 %.
- Based on less than 12 reviews, the Xbox One version scored below 77.5 %.

Original: Sydney to the Max features Sydney Reynolds ( School of Rock ) , Ian Reed Kesler , ( Suits , Kickin ' It ) , Christian J. Simon ( The Amazing World of Gumball ) , Ava Kolker ( Girl Meets World ) , Caroline Rhea ( Sabrina the Teenage Witch , Phineas and Ferb ) , and Jackson Dollinger ( Puppy Dog Pals ) .
Decomposed:
- Sydney to the Max features Sydney Reynolds ( School of Rock ) , Ian Reed Kesler , ( Suits , Kickin ' It ) , Christian J. Simon ( The Amazing World of Gumball ) , Ava Kolker ( Girl Meets World ) , Caroline Rhea ( Sabrina the Teenage Witch , Phineas and Ferb ) , and Jackson Dollinger ( Puppy Dog Pals ) .

Original: {}
Decomposed:
"""
