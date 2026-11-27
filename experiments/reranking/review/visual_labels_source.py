# Each line is confirmed visible entities | visually unresolved entities | observation.
# Authored after viewing image sheets, without using reference captions as ingredient truth.
LABELS='''mushroom garlic herb | | sliced mushrooms and chopped pale/green garnish
rice goji egg | dumpling | fermented-looking rice grains, red goji and yellow egg fragments
bread tomato onion leafy_green | basil | red/yellow tomatoes, bread cubes, red onion and leaves
cake chocolate | marshmallow cream | surface chocolate; interior not visible
pancake banana sauce milk | chocolate | banana and dark sauce, pale drink in background
ice_cream | mango pomelo apple cream | yellow puree underneath white scoop; fruit not identifiable
corn_chip avocado tomato onion herb sauce | cheese | chips, green dip, red diced salsa and orange dip
orange carrot juice | | whole oranges and carrots alongside drink
meat | pork beef sauce | grilled racks; animal species cannot be established
leafy_green egg bread | chicken cheese | lettuce, egg halves and croutons; protein cubes unresolved
potato | meat butter | mashed potato topping conceals interior
ice_cream coffee_bean | coffee | brown frozen scoop with coffee-bean garnish
cake | radish rice shrimp | opaque pan-fried slices; ingredients cannot be identified
cake cream cocoa | cheese coffee | layered sponge with white filling and cocoa dusting
rice sausage pepper herb | chicken shrimp tomato | sausage slices, rice and colored peppers
cake raspberry mint | lemon butter | yellow cake with raspberries and mint
bread tomato herb | cream | red puree and whole tomato alongside, croutons and herb sprig
rice pea | cheese butter | rice grains and green peas
carrot tomato cabbage | potato tofu zucchini | diced vegetables in red broth, pale cubes ambiguous
rice sausage shrimp celery chive | chicken tomato | visible shrimp, sausage slices, diced green stalks
pasta chicken carrot parsley | celery | broad egg-noodle shapes, pale meat pieces and carrots
cake cream cherry | | cherries over white frosting
cookie icing chocolate | | decorated gingerbread-shaped cookies and chocolate pieces
pastry cream chocolate coffee | | cream-filled pastry, dark glaze and coffee beside it
pastry sausage | egg | cut sausage roll exposes cylindrical filling
rice bean meat carrot | goat beef | meat stew beside rice and dark beans; overlay ignored for species
pastry | egg banana rice | folded fried sheets; filling hidden
cucumber dill | yogurt cream | green puree with cucumber slices and dill
rice meat pea corn carrot pepper | beef tomato | minced meat with rice and vegetables
pastry sugar | | fried pastry dusted with sugar
bread bacon onion | pork | bacon and onion on dark bread
sauce parsley | squid onion yogurt cream | breaded rings; interior species unresolved
potato leafy_green | cabbage kale spinach butter | mash with chopped greens
 | | NONFOOD: text says polymer clay tutorial; photographed cake model
noodle carrot bean_sprout meat | chicken cabbage | takeaway fried noodles with strips of vegetables/meat
wonton sauce | pork shrimp cheese | fried wonton shells, red dipping sauce; filling invisible
meat scallion onion | beef pork | sliced browned meat with scallion stalks
cake sugar | cheese egg | pale cake/tart interior cannot establish cheese or egg
meat coriander lemon | chicken beef | cubed meat in sauce; species unresolved
meat coriander | beef chicken | dark stew with fresh coriander
pastry chili sauce | potato egg onion | cut pastry with yellow filling, green chili rings
breadcrumb | pasta cheese shrimp crab | casserole top with crumbs; interior obscured
cake chocolate cream | walnut peanut | chocolate cake/pie, white topping and brown crumbs
cake | taro bean | purple translucent slices; ingredients unresolved
 tomato rice parsley | cheese | stuffed tomatoes with white grain-like filling and parsley
mushroom carrot coriander squash meat | chicken tofu | pale soup, mushrooms and orange slices
bread meat lettuce carrot tomato sauce | pork cheese | folded flatbread with meat and salad
cookie ice_cream | chocolate nut cream | frozen white filling between cookies
strawberry ice_cream | cream coconut | pink frozen dessert topped with strawberry and white flakes
rice | | plain cooked rice
 | apple oat cinnamon sugar | baked crumble with pale fruit cubes; identity uncertain
chicken cucumber lettuce lemon sauce | chili | roasted bird portion with salad and dark condiment
pastry scallion sauce | cheese cream egg | pastry cups with pale filling and green garnish
bread herb | pumpkin carrot cream | yellow puree soup, bread with white spread
croquette lettuce cucumber tomato | potato meat | breaded patties; interior unseen
peanut chili lime rice | papaya cabbage bean | shredded pale salad; main vegetable unresolved
bread ham cheese mustard meat | chicken pork | layered sandwich; pink ham and melted cheese visible
crab coriander | | crab shell and legs with coriander
shrimp noodle tofu scallion mushroom | chicken | prawn soup with noodle strands, tofu and mushroom
meat sauce | beef chicken | stew meat; species not inferable
omelet shrimp mushroom onion pepper herb | cheese | seafood omelet topped with visible shrimp and mushrooms
meat mushroom onion herb | beef tomato | meat and sliced mushrooms in brown/red stew
cookie | butter sugar | pale unbaked cookie dough; invisible ingredients not confirmed
pastry | cherry berry sugar | lattice pie with red fruit filling
fritter sauce | potato onion chickpea mint | orange fritters and two dips, interior hidden
egg bean potato tortilla | meat | fried eggs on flat tortillas beside beans and potatoes
corn bean butter | | corn and pale green beans, butter cube
pork peanut scallion sauce | | skin-on braised pork pieces and peanuts
 dumpling | taro sweet_potato coconut | colored chewy balls/tubers in sweet soup, species uncertain
bread meat onion sauce | pork beef | sandwich with sauce-coated meat and onion
chicken | | fried chicken montage, ingredients beyond chicken unseen
fish cucumber radish wasabi leafy_green | tuna salmon seaweed | sashimi assortment with radish and cucumber; fish species unresolved
fish rice | eel | glazed fish fillets on rice
doughnut icing chocolate | berry | stacked doughnuts with colored glazes
pork leafy_green mustard | spinach | roasted pork slices with greens and mustard
meat scallion | tofu egg pork | minced-looking steamed dish; tofu/egg unresolved
 | broccoli spinach pea cream | green pureed soup without identifiable vegetable
blackberry pastry | custard egg | dark berries in baked tart
bread herb | garlic butter | toasted bread with small green flakes
potato celery onion bacon sauce | mayonnaise cream yogurt cheese | creamy chopped salad
meat sauce | lamb goat | roasted rack/chops, species not established
meat chili peanut scallion coriander sauce | beef pork | red-sauce meat slices with peanuts and herbs
potato | cheese butter onion | baked mashed potato topping
mushroom scallion breadcrumb pepper | cheese sausage | stuffed mushroom caps with crumbs and red/green garnish
meat sauce coriander | beef chicken | meat stew, species unresolved
meatball potato parsley sauce | beef pork | meatballs on potato mash
noodle bean_sprout scallion meat | beef onion | flat stir-fried noodles with sprouts and meat
pastry cream lemon | egg meringue | white-topped lemon-colored tart, lemon slice alongside
noodle tomato peanut carrot leafy_green | bean cucumber | shredded salad with noodle strands
shrimp scallion sauce | | whole shell-on prawns
bread | carrot banana nut | sliced dense loaf; fine inclusions not identified
meat coriander sauce | chicken tomato cream butter | orange curry with meat; exact base invisible
strawberry ice_cream | | frozen strawberry dessert; dairy vs sorbet not distinguished
pastry | egg cream | browned sealed pastry with no visible filling
bread egg olive pepper melon grape | cheese kiwi | egg toast beside mixed fruit bowls
cake chocolate cream sugar | | cake with chocolate frosting and colored sprinkles
cucumber tomato chickpea cheese parsley onion | | vegetable salad with crumbled white cheese
bread shrimp scallion sauce cucumber | | shrimp on toast, cucumber at lower edge
bread sausage mustard sauce | cheese | hot dogs in buns with yellow/red condiments
meat leafy_green chili scallion | liver sausage fish tofu | soup with sliced offal/protein and greens; protein kinds unresolved'''
