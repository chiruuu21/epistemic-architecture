"""
Hand-written evaluation set for belief retrieval and relevance.

    PAIRS        50 (belief, paraphrase): the paraphrase restates the
                 belief while avoiding its content words, so word overlap
                 cannot carry retrieval
    DISTRACTORS  50 beliefs, one per pair, same domain but a different
                 (not contradicting) fact -- stored, never queried for
    UNRELATED    50 prompts from domains with nothing stored (cooking,
                 clothing, household chores, gardening, vehicles, office
                 routine, crafts, furniture, holidays, travel logistics)

10 stored domains x 5: geography, physical science, animals, history,
literature/arts, sports, finance, medicine, weather, computing.

The 12-pair set in smoke_test_sae_semantics.py is the original smoke
test and is kept separate; this one exists so retrieval and relevance
numbers come with usable confidence intervals.
"""

PAIRS = [
    # geography
    ("Paris is the capital of France.",
     "The French government is seated in the city on the Seine."),
    ("The Nile is the longest river in Africa.",
     "No other African waterway runs as far as this Egyptian one."),
    ("Mount Kilimanjaro is in Tanzania.",
     "The tallest African peak rises in an East African nation south of Kenya."),
    ("Australia is both a country and a continent.",
     "That southern landmass is a single nation as well as one of the seven major land areas."),
    ("Tokyo is the largest city in Japan.",
     "No other Japanese metropolis has as many residents as its capital."),
    # physical science
    ("Water boils at one hundred degrees Celsius at sea level.",
     "At normal atmospheric pressure H2O turns to steam once it reaches 100C."),
    ("Light travels faster than sound.",
     "You see the lightning flash before you hear the thunder."),
    ("Iron rusts when exposed to oxygen and moisture.",
     "Leave a steel nail out in damp air and it corrodes reddish brown."),
    ("Gravity pulls objects toward the Earth.",
     "A dropped apple falls down because the planet attracts it."),
    ("Salt dissolves in water.",
     "Stir sodium chloride into a glass and the crystals disappear."),
    # animals
    ("Dogs are loyal pets that many families keep.",
     "Lots of households own a faithful canine companion."),
    ("Bees make honey from flower nectar.",
     "Hive insects turn the sweet liquid they gather from blossoms into a golden syrup."),
    ("Whales are mammals that breathe air.",
     "These huge ocean creatures nurse their young and must surface to inhale."),
    ("Penguins cannot fly.",
     "Those Antarctic birds are unable to take to the air."),
    ("Cheetahs are the fastest land animals.",
     "No creature on the ground can outrun this spotted big cat."),
    # history
    ("World War Two ended in 1945.",
     "The global conflict against the Axis powers was over by the middle of that decade."),
    ("The Roman Empire built roads across Europe.",
     "Legions from the ancient Italian state paved routes throughout the continent."),
    ("The Berlin Wall fell in 1989.",
     "The barrier dividing the German capital came down near the end of the Cold War."),
    ("Neil Armstrong walked on the Moon in 1969.",
     "The Apollo 11 commander became the first person to step onto the lunar surface."),
    ("The printing press was invented by Gutenberg.",
     "A fifteenth century German goldsmith created movable type for producing books."),
    # literature and arts
    ("Shakespeare wrote Hamlet.",
     "The tragedy about the Danish prince came from the Bard of Avon."),
    ("Leonardo da Vinci painted the Mona Lisa.",
     "The famous smiling portrait in the Louvre is the work of a Renaissance polymath."),
    ("Beethoven composed nine symphonies.",
     "The deaf German composer completed nine large orchestral works."),
    ("She plays the violin in an orchestra.",
     "She is a string musician in a symphony ensemble."),
    ("Jane Austen wrote Pride and Prejudice.",
     "The novel about Elizabeth Bennet and Mr Darcy came from an English author of the Regency era."),
    # sports
    ("The team won the championship game last night.",
     "Yesterday evening the squad clinched the title."),
    ("Soccer is the most popular sport in the world.",
     "Football draws more fans worldwide than any other game."),
    ("A marathon is about forty two kilometers long.",
     "That long distance running race covers just over twenty six miles."),
    ("Serena Williams won many tennis titles.",
     "The American racket champion collected numerous Grand Slam trophies."),
    ("The Olympics are held every four years.",
     "The international games come around once every quadrennial cycle."),
    # finance
    ("The stock market fell sharply on Monday.",
     "Share prices dropped hard at the start of the week."),
    ("Inflation reduces the value of money.",
     "As prices rise, each dollar buys less than before."),
    ("The central bank raised interest rates.",
     "Policymakers made borrowing more expensive by lifting the benchmark cost of loans."),
    ("The company reported record profits this quarter.",
     "The firm earned more in these three months than ever before."),
    ("Bonds pay investors a fixed interest.",
     "Debt securities give their holders a set coupon payment."),
    # medicine
    ("The patient was given antibiotics for the infection.",
     "Doctors prescribed medicine to fight his bacterial illness."),
    ("Vaccines train the immune system to fight disease.",
     "Shots teach the body's defenses to recognize pathogens."),
    ("Smoking causes lung cancer.",
     "Tobacco use leads to malignant tumors in the chest."),
    ("The heart pumps blood through the body.",
     "This muscular organ circulates the red fluid through our veins and arteries."),
    ("Regular exercise improves health.",
     "Working out often makes people fitter and less prone to illness."),
    # weather and nature
    ("Heavy rain caused flooding in the town.",
     "Storms left streets underwater after the downpour."),
    ("Hurricanes form over warm ocean water.",
     "Tropical cyclones gather strength above heated seas."),
    ("Snow fell across the mountains overnight.",
     "By morning the peaks were covered in fresh white powder."),
    ("A drought dried up the farmland.",
     "Months without rain left the fields parched and cracked."),
    ("Lightning struck the old oak tree.",
     "A bolt from the storm hit the ancient tree in the yard."),
    # computing
    ("The belief store keeps all entries on the CPU.",
     "Every record in the memory lives in main processor RAM, not graphics hardware."),
    ("The encoder is frozen during training.",
     "Its weights stay fixed and the optimizer never updates them."),
    ("Python is a popular programming language.",
     "Many developers write their code in this widely used scripting tongue."),
    ("The server crashed because it ran out of memory.",
     "The machine went down after exhausting its available RAM."),
    ("Passwords should be long and unique.",
     "Use a lengthy secret phrase that you never reuse across accounts."),
]

DISTRACTORS = [
    # geography
    "Berlin is the capital of Germany.",
    "The Amazon flows through Brazil.",
    "Mount Everest is in Nepal.",
    "Canada is the second largest country by area.",
    "Osaka is famous for its street food.",
    # physical science
    "Water freezes at zero degrees Celsius.",
    "Sound cannot travel through a vacuum.",
    "Copper conducts electricity well.",
    "Magnets attract iron objects.",
    "Oil and water do not mix.",
    # animals
    "Cats are independent pets that many people keep.",
    "Ants live in large colonies.",
    "Sharks have skeletons made of cartilage.",
    "Parrots can imitate human speech.",
    "Elephants have excellent memories.",
    # history
    "World War One began in 1914.",
    "The Egyptians built the pyramids at Giza.",
    "The Soviet Union dissolved in 1991.",
    "Yuri Gagarin was the first human in space.",
    "The telephone was invented by Bell.",
    # literature and arts
    "Charles Dickens wrote Oliver Twist.",
    "Vincent van Gogh painted Starry Night.",
    "Mozart wrote many operas.",
    "He plays the drums in a rock band.",
    "Mark Twain wrote about the Mississippi river.",
    # sports
    "The team signed a new coach last week.",
    "Basketball was invented in Massachusetts.",
    "A golf course usually has eighteen holes.",
    "Roger Federer retired from professional tennis.",
    "The World Cup trophy is made of gold.",
    # finance
    "Gold prices rose on Friday.",
    "Taxes fund public services.",
    "The bank opened a new branch downtown.",
    "The company hired a new chief executive.",
    "Credit cards charge high interest on unpaid balances.",
    # medicine
    "The patient was given painkillers after surgery.",
    "Fevers are a common symptom of the flu.",
    "Sunscreen protects skin from burns.",
    "The kidneys filter waste from the blood.",
    "Sleep is important for memory.",
    # weather and nature
    "Heavy snow closed the roads in the city.",
    "Tornadoes are common in the American Midwest.",
    "Fog covered the harbor this morning.",
    "Volcanoes release ash into the sky.",
    "Wind turned the blades of the windmill.",
    # computing
    "The belief store uses cosine similarity for retrieval.",
    "The decoder weights are updated by gradient descent.",
    "JavaScript runs in web browsers.",
    "The database was backed up last night.",
    "Two factor authentication adds an extra login step.",
]

UNRELATED = [
    # cooking
    "The recipe calls for two cups of flour.",
    "Simmer the onions until they turn soft and golden.",
    "She baked a chocolate cake for dessert.",
    "Add a pinch of pepper before serving the soup.",
    "The bread needs to rise for an hour.",
    # clothing
    "He bought a new pair of leather shoes.",
    "This sweater shrank in the wash.",
    "She wore a red scarf to the party.",
    "The jacket has two deep pockets.",
    "My favorite jeans have a hole in the knee.",
    # household chores
    "I need to vacuum the living room carpet.",
    "The dishes are piling up in the sink.",
    "He folded the laundry on the bed.",
    "Remember to take out the trash on Tuesday.",
    "She dusted the shelves in the hallway.",
    # gardening
    "Plant the basil seedlings in small pots.",
    "Prune the rose bushes in early spring.",
    "The lawn needs mowing again.",
    "She planted tulip bulbs along the fence.",
    "Compost makes the soil richer.",
    # vehicles
    "The car needs new brake pads.",
    "He filled the tank before the long drive.",
    "My bicycle chain keeps slipping.",
    "The mechanic rotated the tires.",
    "She parked the van in the garage.",
    # office routine
    "The weekly meeting ran thirty minutes over.",
    "Please print three copies of the agenda.",
    "The printer on the second floor is jammed.",
    "He scheduled the call for Thursday afternoon.",
    "Her inbox has two hundred unread messages.",
    # crafts and hobbies
    "She learned to knit from her grandmother.",
    "He glued the model airplane together.",
    "The puzzle has a thousand pieces.",
    "They painted pottery at the studio.",
    "I am sewing a quilt for my niece.",
    # furniture and home
    "The sofa is too big for this room.",
    "We assembled the bookshelf ourselves.",
    "The kitchen chair has a wobbly leg.",
    "They bought a new mattress last month.",
    "The lamp by the bed needs a new bulb.",
    # holidays and celebrations
    "We decorated the tree with lights.",
    "The children carved pumpkins for Halloween.",
    "She wrapped the birthday presents in blue paper.",
    "They sent invitations for the wedding.",
    "Fireworks lit up the sky on New Year's Eve.",
    # travel logistics
    "The train to Boston leaves at noon.",
    "Our flight was delayed by two hours.",
    "He packed his suitcase the night before.",
    "The hotel room overlooks the parking lot.",
    "She renewed her passport before the trip.",
]

assert len(PAIRS) == len(DISTRACTORS) == len(UNRELATED) == 50
