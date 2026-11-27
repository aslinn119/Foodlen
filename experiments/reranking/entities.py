"""Explicit ingredient noun matcher. No recipe inference from opaque dish names."""
import re
# Multiword names are consumed before shorter overlapping names.
ALIASES={
'noodle':['noodle','noodles','rice noodles','rice noodle','rice vermicelli','vermicelli','pasta'],
'meat':['meat','meats','beef','veal','mutton','lamb','goat','pork','chicken','oxtail','oxtails','beef tongue','pork belly','beef head skin','meatball','meatballs'],
'bacon':['bacon'], 'ham':['ham'], 'sausage':['sausage','sausages','hot dog','hot dogs','bangers','corn dog','corn dogs'],
'fish':['fish','salmon','tuna','eel','unagi','gravlax','fish head'],
'herb':['herb','herbs','parsley','cilantro','coriander','basil','dill','thyme','oregano','mint','mint leaves','celery leaves'],
'leafy_green':['lettuce','spinach','kale','green leaves','lettuce leaves'],
'pepper':['bell pepper','bell peppers','green pepper','green peppers','red pepper','red peppers','yellow bell peppers'],
'chili':['chili','chilies','chilli','chillies','chili pepper','chili peppers','chili oil'],
'scallion':['scallion','scallions','green onion','green onions','spring onion','spring onions'],
'onion':['onion','onions','onion strips','onion slices','onion rings'],
'ice_cream':['ice cream','icecream','sorbet','sundae'],
'sauce':['sauce','sauces','gravy','salad dressing','dressing','ketchup','soy sauce','mayonnaise'],
'bean_sprout':['bean sprouts','bean sprout'], 'bean':['bean','beans','red beans','red bean','refried beans','green beans','lima beans'],
'corn_chip':['corn chips','corn chip','tortilla chips','nachos','chips'],
'pastry':['pastry','pastries','pie','pies','strudel','tart','tarts','cobbler','crust'],
'cake':['cake','cakes'], 'cookie':['cookie','cookies','biscuit','biscuits'],
'pancake':['pancake','pancakes'], 'doughnut':['doughnut','doughnuts','donut','donuts'],
'cream':['cream','whipped cream','whipping cream'],
'icing':['icing','frosting'], 'sugar':['sugar','sugar powder','powdered sugar','sugar powder','sprinkles'],
'coffee_bean':['coffee bean','coffee beans'], 'coffee':['coffee'],
'apple':['apple','apples','applesauce','apple sauce'],
'egg':['egg','eggs','huevos','omelet','omelette'],
'shrimp':['shrimp','shrimps','prawn','prawns'], 'crab':['crab','crabs','crabmeat'],
'bread':['bread','breads','bun','buns','bread roll','bread rolls','crouton','croutons','toast'],
'sesame':['sesame','sesame seeds','sesame seed'],
'chocolate':['chocolate','chocolate beans'],
'cocoa':['cocoa'], 'nut':['nut','nuts','walnut','walnuts','almond','almonds'],
'peanut':['peanut','peanuts'], 'goji':['goji','medlar','wolfberry','goji berries'],
'chickpea':['chickpea','chickpeas'],
'cabbage':['cabbage','cabbages','purple cabbage'],
'broccoli':['broccoli'],
'garlic':['garlic'], 'ginger':['ginger'], 'tofu':['tofu','aburaage','stinky tofu'],
'juice':['juice'],
}
for key,forms in {
'rice':['rice'],'mushroom':['mushroom','mushrooms'],'tomato':['tomato','tomatoes'],
'carrot':['carrot','carrots'],'potato':['potato','potatoes'],'pea':['pea','peas'],
'cucumber':['cucumber','cucumbers'],'lemon':['lemon','lemons'],'lime':['lime','limes'],
'banana':['banana','bananas'],'orange':['orange','oranges'],'strawberry':['strawberry','strawberries'],
'raspberry':['raspberry','raspberries'],'blueberry':['blueberry','blueberries'],'blackberry':['blackberry','blackberries'],
'cherry':['cherry','cherries'],'avocado':['avocado','avocados'],'pumpkin':['pumpkin'],
'celery':['celery'],'corn':['corn','corn kernels'],'cheese':['cheese'],'butter':['butter'],
'milk':['milk'],'marshmallow':['marshmallow','marshmallows'],'yogurt':['yogurt','yoghurt'],
'radish':['radish','radishes','turnip','turnips'],'squash':['squash'],'taro':['taro'],'sweet_potato':['sweet potato','sweet potatoes'],
'mango':['mango','mangoes'],'pomelo':['pomelo'],'lentil':['lentil','lentils'],
'raisin':['raisin','raisins'],'coconut':['coconut'],'tortilla':['tortilla','tortillas'],
'papaya':['papaya'],'wasabi':['wasabi'],'mustard':['mustard'],'oat':['oat','oats'],
'flour':['flour'],'breadcrumb':['breadcrumb','breadcrumbs'],'melon':['melon','cantaloupe'],
'grape':['grape','grapes'],'olive':['olive','olives'],'seaweed':['seaweed'],
'croquette':['croquette','croquettes'],'fritter':['fritter','fritters'],'wonton':['wonton','wontons','rangoons'],
'dumpling':['dumpling','dumplings','taro balls'],'cookie':['cookie','cookies','biscuit','biscuits'],
'cinnamon':['cinnamon'],'liver':['liver']}.items():ALIASES.setdefault(key,forms)
# Dessert wrappers / whole dish shapes are excluded from primary ingredient analysis.
EXCLUDED={'cake','pastry','cookie','pancake','doughnut','croquette','fritter','wonton','dumpling','juice','icing'}
FORM_TO_ENTITY={f:k for k,fs in ALIASES.items() for f in fs}
PATTERN=re.compile(r'(?<![a-z])(?:'+ '|'.join(re.escape(f) for f in sorted(FORM_TO_ENTITY,key=lambda x:(-len(x),x)))+r')(?![a-z])')
def extract(caption):
    caption=re.sub(r'\borange(?=\s+salmon\b)', '', caption.lower())  # color modifier, not an orange fruit claim
    found={FORM_TO_ENTITY[m.group()] for m in PATTERN.finditer(caption.lower())}
    return sorted(found-EXCLUDED)
def canonical_labels(entities):
    # Match the same granularity for truth and captions.
    aliases={'pasta':'noodle','chicken':'meat','pork':'meat','beef':'meat','goat':'meat','lamb':'meat','mint':'herb','dill':'herb','parsley':'herb','coriander':'herb','chive':'scallion','coffee_bean':'coffee_bean','omelet':'egg','meatball':'meat','mayonnaise':'sauce','oxtail':'meat','eel':'fish','salmon':'fish','tuna':'fish','chocolate':'chocolate','walnut':'nut','berry':'berry'}
    return {aliases.get(e,e) for e in entities}-EXCLUDED
