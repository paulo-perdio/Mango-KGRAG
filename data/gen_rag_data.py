import json

# Original, hand-written mango-cultivation passages (not copied from any source)
# standing in for "Chapter 3" of the real research corpus referenced in the paper.
passages = [
    "Mango trees (Mangifera indica) grow best in tropical and subtropical climates with "
    "temperatures between 24 and 30 degrees Celsius. They require a distinct dry season of "
    "two to three months to trigger uniform flowering, which is why Thailand's central plains "
    "are well suited to commercial mango production.",

    "Well-drained sandy loam soil with a pH between 5.5 and 7.5 is ideal for mango cultivation. "
    "Waterlogged or heavy clay soils increase the risk of root rot and should be avoided or "
    "amended with organic matter and raised planting beds before establishing an orchard.",

    "The Nam Dok Mai variety is prized for its sweet flavor and is typically consumed ripe, "
    "while Keow Savoey is a green-eating variety harvested and sold while still unripe and firm, "
    "valued for its crisp texture and tart taste in Thai cuisine.",

    "Ok Rong mango, widely grown in Chanthaburi province, has a distinctive curved shape and "
    "thin skin. It commands premium prices for export due to its rich aroma and smooth, "
    "fiber-free flesh when fully ripened.",

    "Off-season mango production in Thailand commonly relies on a technique called paclobutrazol "
    "soil drenching, locally referred to as 'raat saan'. This plant growth regulator suppresses "
    "vegetative growth and induces flowering outside the natural season, allowing farmers to "
    "supply mangoes to the market year-round.",

    "The oriental fruit fly, known locally as 'malaeng wan thong', is one of the most damaging "
    "pests to mango orchards. Female flies lay eggs beneath the fruit skin, and larvae feeding "
    "inside the flesh cause internal rot, making infested fruit unsellable.",

    "Anthracnose, caused by the fungus Colletotrichum gloeosporioides, is the most economically "
    "significant disease affecting mango in humid climates. It produces dark, sunken lesions on "
    "leaves, flowers, and fruit, and is most severe during the rainy season when spores spread "
    "via water splash.",

    "Mango flowering occurs on terminal panicles that can carry hundreds of small flowers, but "
    "only a small fraction develop into mature fruit. Poor fruit set is often caused by cool, "
    "wet weather during flowering, which reduces pollinator activity and encourages fungal "
    "infection of the blossoms.",

    "Commercial mango orchards in Thailand are typically spaced 6 to 8 meters apart to allow for "
    "canopy growth and mechanized access. Proper pruning after harvest improves light penetration, "
    "reduces pest pressure, and encourages a stronger flush of new growth before the next flowering "
    "cycle.",

    "Export-grade mango orchards must meet strict quality standards including uniform fruit size, "
    "minimal blemishes, and residue-free pesticide use. Fruit destined for international markets "
    "is often bagged individually on the tree several weeks before harvest to protect it from "
    "insect damage and sunburn.",

    "Mango fruit maturity is judged by a combination of days after flowering, shoulder fullness "
    "near the stem, and skin color change. Harvesting too early results in poor flavor development, "
    "while overripe fruit is prone to bruising and has a shorter shelf life during transport.",

    "Post-harvest handling significantly affects mango quality. Hot water treatment at around 46 to "
    "48 degrees Celsius for several minutes is commonly used to control anthracnose and fruit fly "
    "larvae before export, without damaging the fruit's internal tissue.",

    "Fertilization schedules for mango trees typically shift from nitrogen-heavy formulations during "
    "vegetative growth to phosphorus and potassium emphasis before flowering, since excess nitrogen "
    "close to the flowering period can suppress flower induction and promote leafy growth instead.",

    "Local dialect names for mango varieties and orchard practices vary significantly across "
    "Thailand's regions, which historically made it difficult for agricultural extension officers "
    "and buyers from different provinces to communicate precisely about specific cultivars, pests, "
    "or techniques without a shared standardized vocabulary.",
]

with open("/home/claude/mango-kgrag/data/new_rag_data.jsonl", "w", encoding="utf-8") as f:
    for p in passages:
        inner = json.dumps({"text": p}, ensure_ascii=False)
        line = json.dumps({"json": inner}, ensure_ascii=False)
        f.write(line + "\n")

print("Wrote", len(passages), "passages")
