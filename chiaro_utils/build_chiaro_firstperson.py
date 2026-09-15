#!/usr/bin/env python3
"""Build target-specific first-person CHIARO situations with minimal edits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


UTILS_DIR = Path(__file__).resolve().parent
REPO_ROOT = UTILS_DIR.parent
DEFAULT_INPUT = REPO_ROOT / "Chiaro-main" / "data" / "chiaro_test.json"
DEFAULT_OUTPUT = (
    REPO_ROOT / "Chiaro-main" / "data" / "chiaro_test_firstperson.json"
)

Replacement = tuple[str, str]
Rule = tuple[str, str, tuple[Replacement, ...], tuple[Replacement, ...]]


# Each entry is (expected A role, expected B role, A replacements, B replacements).
# Replacements are deliberately contextual: blind name/pronoun substitution would
# corrupt references to the other participant or produce forms such as "I's".
RULES: tuple[Rule, ...] = (
    (
        "Jordan",
        "Maya",
        (("while Jordan sees", "while I see"),),
        (("Maya sees she isn’t", "I see I’m not"),),
    ),
    (
        "Mia",
        "Jay",
        (
            ("their mom", "my mom"),
            ("Mia’s newly", "my newly"),
            ("Mia looks", "I look"),
            ("she built herself", "I built myself"),
        ),
        (
            ("their mom", "my mom"),
            ("while Jay, in the hallway, thinks", "while I, in the hallway, think"),
            ("his bedroom", "my bedroom"),
        ),
    ),
    (
        "Colin",
        "Maya",
        (
            ("Colin’s name", "my name"),
            ("Colin assumes", "I assume"),
            ("he mishandled", "I mishandled"),
        ),
        (("that Maya repaired", "that I repaired"),),
    ),
    (
        "the stepfather",
        "the stepdaughter",
        (
            ("the stepfather installs", "I install"),
            ("while he sees", "while I see"),
            ("she blamed him and his sons", "she blamed me and my sons"),
        ),
        (
            ("the stepdaughter can", "I can"),
            ("her period products", "my period products"),
            ("that she blamed", "that I blamed"),
        ),
    ),
    (
        "Maya",
        "Jordan",
        (("hands Maya", "hands me"),),
        (("tells Jordan", "tells me"), ("issued to him", "issued to me")),
    ),
    (
        "Maya",
        "Ben",
        (("Maya reads that she is", "I read that I am"),),
        (("Ben reads that his application is", "I read that my application is"),),
    ),
    (
        "Mother",
        "Father",
        (
            ("the mother finishes", "I finish"),
            ("her daughter’s", "my daughter’s"),
            ("and leaves it", "and leave it"),
        ),
        (
            ("the father touches", "I touch"),
            ("his fingers", "my fingers"),
            ("and gets sticky", "and get sticky"),
        ),
    ),
    (
        "Eli",
        "Mara",
        (("Eli’s father", "my father"), ("turns to Eli", "turns to me")),
        (("while Mara is", "while I am"),),
    ),
    (
        "Derek",
        "Maya",
        (("while Derek carries in his", "while I carry in my"),),
        (("letting Maya bring in her", "letting me bring in my"),),
    ),
    (
        "the parent",
        "the wheelchair user",
        (("the parent’s child’s", "my child’s"), ("leaving him", "leaving me")),
        (("the wheelchair user’s scan", "my scan"),),
    ),
    (
        "The guest-of-honor younger sibling",
        "The host sister coordinating",
        (
            ("the guest-of-honor sibling lived", "I lived"),
            ("as the sibling stands", "as I stand"),
        ),
        (("the host sister says", "I say"),),
    ),
    (
        "Mara",
        "Ben",
        (("Mara finishes", "I finish"), ("she has been sewing", "I have been sewing")),
        (
            ("Ben’s place", "my place"),
            ("as he has", "as I have"),
            ("and still needs", "and still need"),
        ),
    ),
    (
        "Jordan",
        "Maya",
        (("while Jordan turns", "while I turn"),),
        (("confronting Maya, she stays in her seat", "confronting me, I stay in my seat"),),
    ),
    (
        "The younger brother",
        "The older sister",
        (("their step-dad", "my step-dad"), ("the younger brother arranged", "I arranged")),
        (("their step-dad", "my step-dad"), ("the older sister could", "I could")),
    ),
    (
        "the younger sister",
        "the older sister",
        (("the younger sister thinks", "I think"), ("her boyfriend", "my boyfriend")),
        (
            ("the older sister reads", "I read"),
            ("about her medical situation and asks", "about my medical situation and ask"),
        ),
    ),
    (
        "John’s spouse",
        "John",
        (("his spouse pushes their bowl", "I push my bowl"),),
        (("awards John", "awards me"), ("so John takes", "so I take"), ("his spouse", "my spouse")),
    ),
    (
        "the junior moderator",
        "the senior moderator",
        (("Casey replied", "I replied"),),
        (("Riley opened", "I opened"),),
    ),
    (
        "the returns clerk at",
        "the parent",
        (("the returns clerk is", "I am"),),
        (
            ("the parent with the receipt won’t", "I, with the receipt, won’t"),
            ("the parent already admitted", "I already admitted"),
        ),
    ),
    (
        "Maya",
        "Evan",
        (("Maya has", "I have"), ("by herself", "by myself")),
        (("while Evan reaches", "while I reach"),),
    ),
    (
        "the cautious cousin visiting",
        "the baby’s mother",
        (("the cautious cousin sees", "I see"),),
        (
            ("the baby’s mother turns her phone", "I turn my phone"),
            ("sees she’s only scrolling", "sees I’m only scrolling"),
            ("while the mother sees", "while I see"),
            ("about her baby", "about my baby"),
        ),
    ),
    (
        "Leo",
        "Maya",
        (("to Leo next door, who has", "to me next door, who have"), ("and pounds", "and pound")),
        (("tells Maya", "tells me"), ("she starts", "I start")),
    ),
    (
        "the husband",
        "the wife",
        (("the husband is left", "I am left"),),
        (("the wife’s portion", "my portion"), ("so she pays only for her", "so I pay only for my")),
    ),
    (
        "Maya",
        "Ethan",
        (("hands Maya", "hands me"), ("she’d been waiting", "I’d been waiting")),
        (("while Ethan freezes", "while I freeze"),),
    ),
    (
        "Mark",
        "Lina",
        (("while Mark sees his", "while I see my"),),
        (("Lina’s name", "my name"),),
    ),
    (
        "Eli",
        "Mara",
        (("while Eli has to cancel his", "while I have to cancel my"),),
        (
            ("Mara reads that her company’s", "I read that my company’s"),
            ("so her job", "so my job"),
        ),
    ),
    (
        "the tenant",
        "the downstairs neighbor with",
        (
            ("the tenant hand-installs", "I hand-install"),
            ("and presses it", "and press it"),
            ("the tenant sees", "I see"),
        ),
        (("the downstairs neighbor with a baby worries", "I, with a baby, worry"),),
    ),
    (
        "Evan",
        "Lena",
        (("it says Evan; Evan heads", "it says my name; I head"),),
        (
            ("while Lena, reaching for the microphone, stands", "while I, reaching for the microphone, stand"),
            ("it isn’t her name", "it isn’t my name"),
        ),
    ),
    (
        "the bride-to-be aunt hearing",
        "the 17-year-old dressmaker finishing",
        (("while the aunt hears", "while I hear"),),
        (("the 17-year-old dressmaker thinks of his", "I think of my"),),
    ),
    (
        "the cashier processing the",
        "the teenage girl",
        (("the cashier hesitates", "I hesitate"),),
        (
            ("the teenage girl sets", "I set"),
            ("and presents her dad’s", "and present my dad’s"),
        ),
    ),
    (
        "the newly licensed teen",
        "the retesting teen handed",
        (
            ("hands one teen a PASS form", "hands me a PASS form"),
            ("the teen with PASS texts", "I text"),
        ),
        (
            ("another teen a FAIL form", "me a FAIL form"),
            ("the teen with FAIL argues", "I argue"),
        ),
    ),
    (
        "the grandmother",
        "the parent working from home",
        (("the grandmother’s wiping mistake", "my wiping mistake"),),
        (("the parent re-wipes their daughter", "I re-wipe my daughter"),),
    ),
    (
        "Trevor",
        "Maya",
        (("Trevor’s deck", "my deck"), ("exposing his falsified", "exposing my falsified")),
        (("Maya’s working", "my working"), ("her months", "my months")),
    ),
    (
        "the woman",
        "the man",
        (("to the late-arriving woman", "to me"),),
        (("leaving the next man at the counter", "leaving me at the counter"),),
    ),
    (
        "the other parent waiting",
        "the kit-making parent",
        (("another parent at the desk pulls back", "I, at the desk, pull back"),),
        (
            ("the parent who assembled a labeled sunscreen-permission packet for their child points", "I, who assembled a labeled sunscreen-permission packet for my child, point"),
        ),
    ),
    (
        "Marco",
        "Lena",
        (("Marco’s omelet", "my omelet"), ("his plate", "my plate"), ("his fingers", "my fingers")),
        (("Lena’s omelet", "my omelet"), ("her pan", "my pan")),
    ),
    (
        "the front-of-line buyer with",
        "the security-delayed customer",
        (("to the front-of-line buyer", "to me"),),
        (
            ("the security-delayed customer arrives", "I arrive"),
            ("and argues", "and argue"),
        ),
    ),
    (
        "the wife on a video call",
        "the husband",
        (
            ("with the wife still on", "with me still on"),
            ("her sister", "my sister"),
            ("her “seven times” claim", "my “seven times” claim"),
        ),
        (("the husband shows", "I show"), ("so he drops", "so I drop")),
    ),
    (
        "the hopeful visitor waiting",
        "the worried visitor",
        (("waves them into", "waves us into"), ("the hopeful visitor expects", "I expect")),
        (("waves them into", "waves us into"), ("the worried visitor thinks", "I think")),
    ),
    (
        "Jay",
        "Mara",
        (
            ("Jay realizes", "I realize"),
            ("his weekend visits", "my weekend visits"),
            ("and may be expected", "and I may be expected"),
        ),
        (("Mara sees", "I see"), ("her work-from-home", "my work-from-home")),
    ),
    (
        "Meg",
        "Meg’s roommate",
        (
            ("Meg finishes", "I finish"),
            ("her effort", "my effort"),
            ("her roommate’s planned", "my roommate’s planned"),
        ),
        (("her roommate’s planned", "my planned"),),
    ),
    (
        "the coworker with the",
        "the graphic designer",
        (("her coworker’s last remaining", "my last remaining"),),
        (
            ("the graphic designer gets", "I get"),
            ("she stayed late", "I stayed late"),
            ("her coworker’s last remaining", "my coworker’s last remaining"),
        ),
    ),
    (
        "the mother",
        "the daughter",
        (("while her mother stares", "while I stare"), ("and thinks", "and think")),
        (
            ("the daughter takes it and marks", "I take it and mark"),
            ("her mother", "my mother"),
        ),
    ),
    (
        "the musician neighbor",
        "the next-door neighbor",
        (("the musician neighbor rehearses", "I rehearse"), ("their first", "my first")),
        (("as the next-door neighbor sits", "as I sit"), ("their box", "my box")),
    ),
    (
        "the performer with a",
        "the stage manager",
        (("another performer’s costume", "my costume"),),
        (("the stage manager hung", "I hung"),),
    ),
    (
        "the booth owner",
        "the rival booth owner",
        (("one booth owner’s sign", "my sign"),),
        (("the other booth owner’s sign", "my sign"),),
    ),
    (
        "Adam",
        "Mark",
        (
            ("After Adam carries", "After I carry"),
            ("they set", "we set"),
            ("and Adam studies", "and I study"),
            ("and dwells", "and dwell"),
        ),
        (
            ("for Mark when Mark’s knee", "for me when my knee"),
            ("they set", "we set"),
            ("Mark expresses", "I express"),
        ),
    ),
    (
        "the date",
        "the customer on a date",
        (
            ("their water", "our water"),
            ("their tightly packed table", "our tightly packed table"),
            ("his date sees", "I see"),
        ),
        (
            ("their water", "our water"),
            ("their tightly packed table", "our tightly packed table"),
            ("so he can commend", "so I can commend"),
            ("his date", "my date"),
        ),
    ),
    (
        "Maya",
        "Jordan",
        (("Maya sets her", "I set my"),),
        (
            ("Jordan finds he has", "I find I have"),
            ("his small fan", "my small fan"),
            ("his night shift", "my night shift"),
        ),
    ),
    (
        "Maya",
        "Ethan",
        (("incorrectly by Maya", "incorrectly by me"),),
        (("announces Ethan’s pie", "announces my pie"),),
    ),
    (
        "Lina",
        "Mark",
        (("shows Lina how to crimp her", "shows me how to crimp my"),),
        (("Mark’s bowl", "my bowl"),),
    ),
    (
        "the first contestant freed",
        "the next contestant left",
        (("the first contestant is released", "I am released"),),
        (("the next contestant is left", "I am left"),),
    ),
    (
        "Lisa",
        "Maya",
        (("the replacement associate signs", "I sign"),),
        (("Maya’s upcoming shifts", "my upcoming shifts"),),
    ),
    (
        "the other home baker",
        "the home baker waiting",
        (
            ("the other entrant’s pie", "my pie"),
            ("leaving him", "leaving me"),
        ),
        (("the first-time entrant’s pie", "my pie"),),
    ),
    (
        "Nina",
        "Mara",
        (("Nina’s first attempt", "my first attempt"), ("before she can", "before I can")),
        (("Mara, after weeks of practice, sets her", "I, after weeks of practice, set my"),),
    ),
    (
        "Evan",
        "Maya",
        (
            ("Evan’s careful questions", "my careful questions"),
            ("while Evan pictures", "while I picture"),
            ("leaving him", "leaving me"),
        ),
        (
            ("in Maya’s name", "in my name"),
            ("keep her from signing", "keep me from signing"),
            ("before she understands", "before I understand"),
        ),
    ),
    (
        "Evan",
        "Maya",
        (
            ("and Evan stands beside his", "and I stand beside my"),
            ("sees his entry", "sees my entry"),
        ),
        (("Maya’s bridge", "my bridge"),),
    ),
    (
        "Maya",
        "Alex",
        (
            ("Maya’s daffodils", "my daffodils"),
            ("Maya’s favorite dish", "my favorite dish"),
            ("Maya leaves", "I leave"),
        ),
        (("Alex’s mother", "my mother"), ("while Alex recoils", "while I recoil")),
    ),
    (
        "the hiring manager conducting",
        "the 19-year-old internship applicant",
        (("During the Zoom interview", "During the Zoom interview I am conducting"),),
        (
            ("the applicant completes", "I complete"),
            ("on his own", "on my own"),
            ("explains his code", "explain my code"),
            ("behind him", "behind me"),
            ("his mother", "my mother"),
            ("for him", "for me"),
        ),
    ),
    (
        "the soft-spoken in-law",
        "the cancer patient",
        (
            ("the in-law’s comforting words", "my comforting words"),
            ("but the in-law becomes", "but I become"),
            ("what she said", "what I said"),
        ),
        (("to the cancer patient", "to me"), ("the cancer patient can", "I can")),
    ),
    (
        "Carmen",
        "Carmen’s sister-in-law",
        (
            ("addresses Carmen by her name", "addresses me by my name"),
            ("Carmen no longer has", "I no longer have"),
            ("Carmen’s sister-in-law", "my sister-in-law"),
        ),
        (("Carmen’s sister-in-law has", "I have"),),
    ),
    (
        "Lena",
        "Ramon",
        (
            ("Lena’s phone", "my phone"),
            ("her biological father", "my biological father"),
            ("and Lena pauses", "and I pause"),
        ),
        (("Ramon’s months", "my months"),),
    ),
    (
        "Eli",
        "Maya",
        (
            ("Eli acknowledges that he", "I acknowledge that I"),
            ("talks down", "talk down"),
            ("while Eli realizes", "while I realize"),
            ("his date", "my date"),
        ),
        (("Maya finally gets", "I finally get"), ("she has waited", "I have waited")),
    ),
    (
        "Maya",
        "Jordan",
        (("hazard Maya had worried", "hazard I had worried"),),
        (("and Jordan, who tripped", "and I, who tripped"), ("becomes the one", "become the one")),
    ),
    (
        "Maya",
        "Derek",
        (("As Maya ties", "As I tie"), ("Maya ends up", "I end up")),
        (("while Derek worries", "while I worry"), ("keep him from", "keep me from")),
    ),
    (
        "Mina",
        "the bride",
        (("Mina now has", "I now have"),),
        (
            ("the bride’s private", "my private"),
            ("while the bride is", "while I am"),
            ("she expected", "I expected"),
        ),
    ),
    (
        "John",
        "Sarah",
        (("John’s teammates", "my teammates"),),
        (("asks Sarah", "asks me"), ("messages she sent", "messages I sent")),
    ),
    (
        "Mara",
        "Eli",
        (("listing Mara as", "listing me as"),),
        (("beside Eli’s name", "beside my name"),),
    ),
    (
        "the unemployed twin",
        "the downstairs neighbor with",
        (("the unemployed twin keeps", "I keep"),),
        (("the downstairs neighbor stares", "I stare"),),
    ),
    (
        "the next-door neighbor with",
        "the 18-year-old younger sibling",
        (("while the next-door neighbor with an early shift lies", "while I, with an early shift, lie"),),
        (
            ("the 18-year-old younger sibling finds", "I find"),
            ("held for them", "held for me"),
        ),
    ),
    (
        "Mina",
        "Evan",
        (
            ("and Mina in the next apartment stays", "and I, in the next apartment, stay"),
            ("because she might", "because I might"),
            ("before her early", "before my early"),
        ),
        (
            ("After Evan finally plays", "After I finally play"),
            ("from his days", "from my days"),
            ("he keeps repeating", "I keep repeating"),
            ("for his community", "for my community"),
        ),
    ),
    (
        "Maya",
        "Linda",
        (
            ("Maya’s strict relatives", "my strict relatives"),
            ("Maya avoids", "I avoid"),
            ("she feared", "I feared"),
        ),
        (("while Linda sees", "while I see"), ("her dry-wedding", "my dry-wedding")),
    ),
    (
        "Jenna",
        "Maya",
        (("that Jenna is safe", "that I am safe"), ("Jenna used", "I used")),
        (
            ("Maya’s phone", "my phone"),
            ("Maya had been bracing", "I had been bracing"),
        ),
    ),
    (
        "Maya",
        "Aiden’s mom",
        (
            ("Maya’s wrist", "my wrist"),
            ("after Maya explains", "after I explain"),
            ("Maya has direct", "I have direct"),
        ),
        (
            ("Aiden’s mom sits", "I sit"),
            ("her son", "my son"),
            ("she can’t keep", "I can’t keep"),
        ),
    ),
    (
        "the student shown in",
        "the club volunteer posting",
        (("includes another student", "includes me"), ("recognize them", "recognize me")),
        (("the club volunteer posts", "I post"),),
    ),
    (
        "Evan",
        "Mara",
        (
            ("When Evan opens", "When I open"),
            ("his $5,000", "my $5,000"),
            ("his case", "my case"),
            ("his holiday", "my holiday"),
        ),
        (("but Mara in hospital marketing sees", "but I, in hospital marketing, see"),),
    ),
    (
        "the self-employed data-network tradesman",
        "the union electrician dad",
        (
            ("the self-employed data-network tradesman stepdad was", "I was"),
            ("his tool bag", "my tool bag"),
        ),
        (("credits the union electrician dad", "credits me"),),
    ),
    (
        "Noah",
        "Liam",
        (
            ("Noah, who paid his", "I, who paid my"),
            ("sees his name", "see my name"),
        ),
        (
            ("while Liam, who never", "while I, who never"),
            ("stands with classmates", "stand with classmates"),
            ("his name missing", "my name missing"),
        ),
    ),
    (
        "Kara",
        "Mia",
        (("while Kara slides", "while I slide"),),
        (("Mia tightens", "I tighten"), ("her wrapped", "my wrapped")),
    ),
    (
        "Mara",
        "Dana",
        (("Mara’s meal", "my meal"), ("Mara leaves", "I leave")),
        (("Dana pulls her hands", "I pull my hands"),),
    ),
    (
        "Maya",
        "Leo",
        (("Maya’s laptop", "my laptop"),),
        (
            ("but Leo sees", "but I see"),
            ("place him", "place me"),
            ("he doesn’t", "I don’t"),
        ),
    ),
    (
        "Maya",
        "Lina",
        (("showing Maya in", "showing me in"),),
        (("and Lina sees", "and I see"), ("so she must", "so I must")),
    ),
    (
        "the unlisted roommate stuck",
        "the leaseholding tenant with",
        (
            ("while the unlisted roommate stands", "while I stand"),
            ("as her bike", "as my bike"),
        ),
        (("the leaseholding tenant locks", "I lock"),),
    ),
    (
        "the guest",
        "the tenant",
        (
            ("her guest is joining the pages with her", "I am joining the pages with her"),
            ("while the guest argues that his", "while I argue that my"),
            ("his show", "my show"),
        ),
        (
            ("the tenant points out", "I point out"),
            ("her guest", "my guest"),
            ("with her", "with me"),
        ),
    ),
    (
        "the guest",
        "the parent with a",
        (("the guest’s urgent", "my urgent"), ("giving the guest", "giving me")),
        (("the parent lifts", "I lift"), ("and swings its", "and swing its")),
    ),
    (
        "the houseguest",
        "the homeowner",
        (
            ("while the houseguest who brought", "while I, who brought"),
            ("the plug-ins worries she’ll", "the plug-ins, worry I’ll"),
        ),
        (("the homeowner no longer expects", "I no longer expect"),),
    ),
    (
        "Maya",
        "Evan",
        (("hands Maya", "hands me"),),
        (("leaving Evan", "leaving me"),),
    ),
    (
        "The neighbor",
        "The homeowner returning to stop a loud",
        (
            ("the next-door neighbor sees", "I see"),
            ("his shrubs", "my shrubs"),
        ),
        (("the homeowner has his house", "I have my house"),),
    ),
    (
        "the husband",
        "the younger brother",
        (("the Zoom audition", "my Zoom audition"), ("so the audition isn’t", "so my audition isn’t")),
        (("while the younger brother has", "while I have"),),
    ),
    (
        "the roommate",
        "the sleepwalking partner",
        (
            ("the roommate whose bedroom", "I, whose bedroom"),
            ("was mistaken now", "was mistaken, now"),
            ("now has a clear", "now have a clear"),
        ),
        (
            ("the sleepwalking partner still faces", "I still face"),
            ("if she wakes", "if I wake"),
        ),
    ),
    (
        "the checked-bag owner",
        "the standby passenger cleared",
        (("the bag’s owner starts", "I start"), ("her clothes", "my clothes")),
        (("the standby passenger is", "I am"),),
    ),
    (
        "the barista working the",
        "Harry",
        (
            ("leaving the barista flustered", "leaving me flustered"),
            ("as she fumbles", "as I fumble"),
        ),
        (("Harry loudly rehearses his", "I loudly rehearse my"),),
    ),
    (
        "The grandparent",
        "Daisy",
        (
            ("the grandparent finishes", "I finish"),
            ("their own cloud", "my own cloud"),
        ),
        (
            ("worrying Daisy will", "worrying I will"),
            ("and Daisy, who is", "and I, who am"),
            ("sees the lock", "see the lock"),
            ("stop her", "stop me"),
        ),
    ),
    (
        "Maya",
        "Lena",
        (("walks Maya and her", "walks me and my"),),
        (("leaving Lena", "leaving me"),),
    ),
    (
        "the bill-splitting housemate",
        "the meticulous housemate",
        (("the other housemate’s Venmo", "my Venmo"),),
        (("one housemate saved", "I saved"),),
    ),
    (
        "the aunt waiting with",
        "the child’s mother",
        (("pulls the aunt and", "pulls me and"),),
        (
            ("stopping the mother", "stopping me"),
            ("telling her she isn’t", "telling me I’m not"),
        ),
    ),
    (
        "Mara",
        "Lina",
        (("hands Mara", "hands me"), ("to Mara", "to me")),
        (("Lina’s wedding", "my wedding"), ("Lina cheated", "I cheated")),
    ),
    (
        "the 17-year-old student",
        "the student's father",
        (("the student clicks", "I click"), ("their father", "my father")),
        (("their father still has", "I still have"),),
    ),
    (
        "Jordan",
        "Casey",
        (("while Jordan backs", "while I back"),),
        (("Casey sees", "I see"),),
    ),
    (
        "Mom",
        "I",
        (
            ("At Mom’s birthday", "At my birthday"),
            ("at Mom’s place", "at my place"),
            ("my novel", "their novel"),
            ("while Mom is left", "while I am left"),
            ("meant for her", "meant for me"),
        ),
        (),
    ),
    (
        "Maya",
        "Leah",
        (
            ("Maya splashes", "I splash"),
            ("and smears makeup", "and smear makeup"),
            ("setting up her", "setting up my"),
        ),
        (("leaving Leah", "leaving me"), ("before she can", "before I can")),
    ),
)


def apply_replacements(
    sentence: str, replacements: tuple[Replacement, ...], *, item_id: str, slot: str
) -> str:
    rewritten = sentence
    for old, new in replacements:
        count = rewritten.count(old)
        if count != 1:
            raise ValueError(
                f"{item_id} slot {slot}: expected one occurrence of {old!r}, "
                f"found {count}"
            )
        rewritten = rewritten.replace(old, new, 1)
    return rewritten


def build_firstperson_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(items) != len(RULES):
        raise ValueError(f"Expected {len(RULES)} CHIARO items, found {len(items)}")
    output: list[dict[str, Any]] = []
    for item, (role_a, role_b, replacements_a, replacements_b) in zip(
        items, RULES, strict=True
    ):
        item_id = str(item.get("id", ""))
        actual_roles = (item.get("agent_a_role"), item.get("agent_b_role"))
        if actual_roles != (role_a, role_b):
            raise ValueError(
                f"{item_id}: role mismatch; expected={(role_a, role_b)!r}, "
                f"actual={actual_roles!r}"
            )
        sentence = item.get("sentence")
        if not isinstance(sentence, str) or not sentence:
            raise ValueError(f"{item_id}: sentence must be non-empty text")
        sentence_a = apply_replacements(
            sentence, replacements_a, item_id=item_id, slot="A"
        )
        sentence_b = apply_replacements(
            sentence, replacements_b, item_id=item_id, slot="B"
        )
        if not replacements_a or sentence_a == sentence:
            if role_a != "I":
                raise ValueError(f"{item_id} slot A: sentence was not rewritten")
        if not replacements_b or sentence_b == sentence:
            if role_b != "I":
                raise ValueError(f"{item_id} slot B: sentence was not rewritten")

        converted: dict[str, Any] = {}
        for key, value in item.items():
            converted[key] = value
            if key == "sentence":
                converted["sentence_a_firstperson"] = sentence_a
                converted["sentence_b_firstperson"] = sentence_b
        output.append(converted)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build target-specific first-person CHIARO test situations"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    with args.input.open("r", encoding="utf-8") as handle:
        items = json.load(handle)
    if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
        raise ValueError("CHIARO input must be a JSON list of objects")
    output = build_firstperson_items(items)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(args.output)
    print(f"[firstperson] wrote {len(output)} scenes to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
