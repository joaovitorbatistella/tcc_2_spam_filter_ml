import joblib
import os

dirname = os.path.dirname(__file__)

model = joblib.load(dirname+'/../output/tfidf_logreg_model.pkl')

# messages = [
#     "Congratulations! Thanks to a good friend U have WON the �2,000 Xmas prize. 2 claim is easy, just call 08718726978 NOW! Only 10p per minute. BT-national-rate",
#     "I will send them to your email. Do you mind  &lt;#&gt;  times per night?",
#     "44 7732584351, Do you want a New Nokia 3510i colour phone DeliveredTomorrow? With 300 free minutes to any mobile + 100 free texts + Free Camcorder reply or call 08000930705.",
#     "tap & spile at seven. * Is that pub on gas st off broad st by canal. Ok?",
#     "Ok then i come n pick u at engin?",
#     "You have 1 new voicemail. Please call 08719181513.",
#     "MOON has come to color your dreams, STARS to make them musical and my SMS to give you warm and Peaceful Sleep. Good Night",
#     "Just finished eating. Got u a plate. NOT leftovers this time.",
#     "Thanx a lot...",
#     "Hurry home u big butt. Hang up on your last caller if u have to. Food is done and I'm starving. Don't ask what I cooked.",
#     "Lol your right. What diet? Everyday I cheat anyway. I'm meant to be a fatty :(",
# ]

messages = [
    """effective 12 - 11 - 99
| - - - - - - - - + - - - - - - - - - - + - - - - - - - - - - - |
| | | |
| mscf / d | min ftp | time |
| | | |
| - - - - - - - - + - - - - - - - - - - + - - - - - - - - - - - |
| | | |
| 4 , 500 | 9 , 925 | 24 hours |
| | | |
| - - - - - - - - + - - - - - - - - - - + - - - - - - - - - - - |
| | | |
| 6 , 000 | 9 , 908 | 24 hours |
| | | |
| - - - - - - - - + - - - - - - - - - - + - - - - - - - - - - - |
| | | |
| 8 , 000 | 9 , 878 | 24 hours |
| | | |
| - - - - - - - - + - - - - - - - - - - + - - - - - - - - - - - |
| | | |
| 10 , 000 | 9 , 840 | 24 hours |
| | | |
| - - - - - - - - + - - - - - - - - - - + - - - - - - - - - - - |
| | | |
| 12 , 000 | 9 , 793 | 24 hours |
| | | |
| - - - - - - - - + - - - - - - - - - - + - - - - - - - - - - - |
| | | |
| 14 , 000 | 9 , 738 | 24 hours |
| | | |
| - - - - - - - - + - - - - - - - - - - + - - - - - - - - - - - |
| | | |
| 16 , 000 | 9 , 674 | 24 hours |
| | | |
| - - - - - - - - + - - - - - - - - - - + - - - - - - - - - - - |
| | | |
| 18 , 000 | 9 , 602 | 24 hours |
| | | |
| - - - - - - - - + - - - - - - - - - - + - - - - - - - - - - - |
| | | |
| 20 , 000 | 9 , 521 | 24 hours |
| | | |
| - - - - - - - - + - - - - - - - - - - + - - - - - - - - - - - |
| | | |
| 22 , 000 | 9 , 431 | 24 hours |
| | | |
| - - - - - - - - + - - - - - - - - - - + - - - - - - - - - - - |
| | | |
| 24 , 000 | 9 , 332 | 24 hours |
| | | |
| - - - - - - - - + - - - - - - - - - - + - - - - - - - - - - - |
| | | |
| 26 , 000 | 9 , 224 | 24 hours |
| | | |
| - - - - - - - - + - - - - - - - - - - + - - - - - - - - - - - |
| | | |
| 28 , 000 | 9 , 108 | 24 hours |
| | | |
| - - - - - - - - + - - - - - - - - - - + - - - - - - - - - - - |
| | | |
| 30 , 000 | 8 , 982 | 24 hours |
| | | |
| - - - - - - - - + - - - - - - - - - - + - - - - - - - - - - - |
| | | |
| 32 , 000 | 8 , 847 | 24 hours |
| | | |
| - - - - - - - - + - - - - - - - - - - + - - - - - - - - - - - |
| | | |
| 34 , 000 | 8 , 703 | 24 hours |
| | | |
| - - - - - - - - + - - - - - - - - - - + - - - - - - - - - - - |
| | | |
| 36 , 000 | 8 , 549 | 24 hours |
| | | |
| - - - - - - - - + - - - - - - - - - - + - - - - - - - - - - - |""",
"""- calpine daily gas nomination 1 . doc""",
"""i have to create accounting arrangement for purchase from unocal energy at
meter 986782 . deal not tracked for 5 / 99 . volume on deal 114427 expired 4 / 99 .""",
"""kim / anita -
a volume of 7247 mm shows to have been allocated to the reliant 201 contract
for november . there was no nomination for reliant at this point in november
and , therefore , there should be no volume allocated to their contract .
please make sure these volumes are moved off the reliant contract prior to
november close .
thanks .""",
"""jackie ,
since the inlet to 3 river plant is shut in on 10 / 19 / 99 ( the last day of
flow ) :
at what meter is the mcmullen gas being diverted to ?
at what meter is hpl buying the residue gas ? ( this is the gas from teco ,
vastar , vintage , tejones , and swift )
i still see active deals at meter 3405 in path manager for teco , vastar ,
vintage , tejones , and swift
i also see gas scheduled in pops at meter 3404 and 3405 .
please advice . we need to resolve this as soon as possible so settlement
can send out payments .
thanks""",
"""george ,
i need the following done :
jan 13
zero out 012 - 27049 - 02 - 001 receipt package id 2666
allocate flow of 149 to 012 - 64610 - 02 - 055 deliv package id 392
jan 26
zero out 012 - 27049 - 02 - 001 receipt package id 3011
zero out 012 - 64610 - 02 - 055 deliv package id 392
these were buybacks that were incorrectly nominated to transport contracts
( ect 201 receipt )
let me know when this is done
hc""",
"""i will be making these changes at 11 : 00 am on wednesday december 15 .
if you do not agree or have a problem with the dnb number change please
notify me , otherwise i will make the change as scheduled .
dunns number change :
counterparty cp id number
from to
cinergy resources inc . 62163 869279893 928976257
energy dynamics management , inc . 69545 825854664 088889774
south jersey resources group llc 52109 789118270 036474336
transalta energy marketing ( us ) inc . 62413 252050406 255326837
philadelphia gas works 33282 148415904 146907159
thanks ,
rennie
3 - 7578""",
"""iris
i would like you to put me in contact with s / one at enron here in london that
deals with weather derivatives and would be in a position to sell us options on
weather derivatives ( temperature , cat ) . let me know if you are able to do that
or if i need to work internally here in order to find out whom we have contacts
with at enron .
if you want to call me my direct line is + 44 207 336 - 2836 . alternatively i could
call you but do bear in mind that i leave the office around 6 : 30 - 7 pm london
time . send me an email and let me know when is a good time to talk and i will
call you back .
thanks in advance .
antonella
this transmission has been issued by a member of the hsbc group ( \"\" hsbc \"\" )
for the information of the addressee only and should not be reproduced
and / or distributed to any other person . each page attached hereto must
be read in conjunction with any disclaimer which forms part of it . unless
otherwise stated , this transmission is neither an offer nor the solicitation
of an offer to sell or purchase any investment . its contents are based on
information obtained from sources believed to be reliable but hsbc makes
no representation and accepts no responsibility or liability as to its
completeness or accuracy .""",
"""iris
i would like you to put me in contact with s / one at enron here in london that
deals with weather derivatives and would be in a position to sell us options on
weather derivatives ( temperature , cat ) . let me know if you are able to do that
or if i need to work internally here in order to find out whom we have contacts
with at enron .
if you want to call me my direct line is + 44 207 336 - 2836 . alternatively i could
call you but do bear in mind that i leave the office around 6 : 30 - 7 pm london
time . send me an email and let me know when is a good time to talk and i will
call you back .
thanks in advance .
antonella
this transmission has been issued by a member of the hsbc group ( \"\" hsbc \"\" )
for the information of the addressee only and should not be reproduced
and / or distributed to any other person . each page attached hereto must
be read in conjunction with any disclaimer which forms part of it . unless
otherwise stated , this transmission is neither an offer nor the solicitation
of an offer to sell or purchase any investment . its contents are based on
information obtained from sources believed to be reliable but hsbc makes
no representation and accepts no responsibility or liability as to its
completeness or accuracy ."""
]

predictions = model.predict(messages)

print([predictions])