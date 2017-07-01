import json
import collections
import pandas as pd
import glob
import sys
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import SGDClassifier
import numpy as np
import random
import datetime
import operator
import html
import google
import newspaper

random.seed(100)
start_time = datetime.datetime.now()

# ---------------------- Reading Json file for Test data

with open('Test//TestData.json', 'r') as content_file:
    json_data = content_file.read()
py_object_test_data = json.loads(json_data, object_pairs_hook=collections.OrderedDict)['TestData']
test_data_x = []

for key, value in py_object_test_data.items():
    test_data_x.append(value["bodyText"])


# ---------------------- Reading Prediction Labels

labels = []
with open('Test//topicDictionary.txt', 'r') as content_file:
    labels = content_file.read().split("\n")
print(labels)

# ---------------------- Reading & Cleaning the Json file for Training set

print("Reading the input data...")
files = glob.glob("*.json")
data = dict
flag = 0
train_data_x = []
train_data_y = []
unique_train_labels = set()
# label_weights = {}
#
# for l in labels:
#     label_weights[l] = 0

for file in files[26:]:
    with open(file, 'r') as content_file:
        json_data = content_file.read()
    py_object_train_data = json.loads(json_data, object_pairs_hook=collections.OrderedDict)['TrainingData']

    print("File " + str(flag + 1))
    flag += 1

    for key, value in py_object_train_data.items():
        if value["topics"] and value["bodyText"] != "":
            temp = []
            for each_label in value["topics"]:
                if each_label in labels:
                    temp.append(each_label)
                    #label_weights[each_label] += 1
            if temp:
                unique_train_labels.update(temp)
                train_data_x.append(value["bodyText"])
                train_data_y.append(temp)
    print(len(train_data_x))
    print(len(train_data_y))
    del py_object_train_data
    del json_data

print("Number of unique labels in Training set:")
print(len(unique_train_labels))
# print(sum(list(label_weights.values())))
# print(np.median(list(label_weights.values())))
# print(np.mean(list(label_weights.values())))
# #print(list(label_weights.values()))
# print(sorted(label_weights.items(), key = operator.itemgetter(1)))
# print(sorted(label_weights.items()))
# for i in label_weights:
#     if label_weights[i]<25:
#         label_weights[i] = 2
#     else:
#         label_weights[i] = 1
#     #label_weights[i] = len(train_data_x)/(len(unique_train_labels) * label_weights[i])
# print(sorted(label_weights.items()))
print("*"*10)

# ---------------------- Determining labels for which there exists no train data
non_existent = list(set(labels).difference(unique_train_labels))
print(non_existent)
# print("Adding data for the missing labels")
# for label in non_existent:
#     cnt = 0
#     search_results = google.search(label + " theguardian", stop=4, lang="en")
#     print("*"*30)
#     for link in search_results:
#         if not "www.theguardian.com/" in link:
#             break
#         data = newspaper.Article(url=link)
#         data.download()
#         data.parse()
#         text = data.text.replace("\n"," ")
#         if text:
#             train_data_y.append([label])
#             train_data_x.append(text)
#             cnt += 1
#         if cnt >= 5:
#             break
#
# print("Done adding data for the missing labels")

# zikavirus
# train_data_y.append(['zikavirus'])
# train_data_x.append(html.unescape("The World Health Organisation has convened an emergency committee to discuss the \u201cexplosive\u201d spread of the Zika virus, which has been linked to thousands of birth defects in Latin America. \u201cLast year the disease was detected in the Americas, where it is spreading explosively,\u201d Margaret Chan, the WHO director general, said at a special briefing in Geneva. It was \u201cdeeply concerning\u201d that the virus had now been detected in 23 countries in the Americas, she added. One WHO scientist estimated there could be 3-4m Zika infections in the Americas over the next year. The spread of the virus has prompted governments across the world to advise pregnant women against going to the areas where it has been detected. There is no vaccine or cure for Zika, which has been linked to microcephaly, a serious condition that can cause lifelong developmental problems. Chan said: \u201cThe level of alarm is extremely high. Arrival of the virus in some cases has been associated with a steep increase in the birth of babies with abnormally small heads.\u201d She added: \u201cA causal relationship between Zika virus and birth malformations and neurological syndromes has not yet been established \u2013 this is an important point \u2013 but it is strongly suspected. \u201cThe possible links have rapidly changed the risk profile of Zika from a mild threat to one of alarming proportions. The increased incidence of microcephaly is particularly alarming as it places a heartbreaking burden on families and communities.\u201d Chan outlined four reasons for alarm: \u201cFirst, the possible association of infection with birth malformations and neurological syndromes. Second, the potential for further international spread given the wide geographical distribution of the mosquito vector. Third, the lack of population immunity in newly affected areas. Fourth, the absence of vaccines.\u201d This year\u2019s El Ni\u00f1o weather patterns meant mosquito populations were expected to spread, Chan added. \u201cFor all these reasons, I have decided to convene an emergency committee under the international health regulations,\u201d she said. The committee will meet on Monday and will advise on the international responses and specific measures in affected countries and elsewhere. Brazilian authorities estimate the country could have up to 1m Zika infections by now, and since September, the country has registered nearly 4,000 cases of babies with microcephaly. The Zika outbreak and spike in microcephaly cases have been concentrated in the poor and underdeveloped north-east. But the south-east, where S\u00e3o Paulo and Rio de Janeiro are located, is the nation\u2019s second hardest-hit region. Rio de Janeiro is of particular concern, since it will host the Olympic games this summer. The president of the International Olympic Committee, Thomas Bach, said the IOC was in \u201cclose contact\u201d with Brazilian authorities and the WHO, and that all national Olympic bodies would be advised on how to deal with the virus before the Games started. The Brazilian president, Dilma Rousseff, has pledged to wage war against the Aedes aegypti mosquito that spreads the virus, focusing on getting rid of the insect\u2019s breeding grounds. The US Centers for Disease Control and Prevention said there had been 31 cases of Zika infection among US citizens who travelled to areas affected by the virus, but so far there had been no cases of transmission of the virus through mosquitoes in the US itself. The White House said its experts were most concerned about its potential impact on women who are pregnant or could become pregnant. US officials said the country had two potential candidates for a vaccine, and might begin clinical trials in people by the end of this year. But experts in disease control have warned they do not expect to have a vaccine available in 2016. Dr Anthony Fauci, director of the National Institute of Allergy and Infectious Diseases, said on Thursday that previous research into dengue fever, the West Nile virus and the chikungunya virus would give scientists an \u201cexisting vaccine platform\u201d which could be used as \u201ca jumping-off point\u201d for finding a cure to the Zika virus. \u201cIt is important to note that we will not have a widely available safe and effective Zika vaccine this year and probably not in the next few years,\u201d Fauci said, before adding that scientists might be able to begin \u201ca phased clinical trial in this calendar year\u201d. Addressing the global threat, Lawrence Gostin, a public health law expert from Georgetown University, warned that Zika had an \u201cexplosive pandemic potential\u201d. Speaking to the BBC\u2019s World Service, Gostin, a member of a commission that criticised the WHO for its response to Ebola, said: \u201cWith the Rio Olympics on our doorstep I can certainly see this having a pandemic potential.\u201d He said every review of the WHO\u2019s response to Ebola found that it was \u201ctoo little, too late\u201d. Interviewed minutes before Chan\u2019s announcement, he said: \u201cI\u2019m disappointed that the WHO has not been acting proactively. They have not issued any advice about travel, about surveillance, about mosquito control. \u201cThe very first thing I would propose is a global mosquito eradication effort, particularly in areas with ongoing Zika transmission. We really need to declare war on this species of mosquito.\u201d The WHO\u2019s leadership admitted last April to serious missteps in its handling of the Ebola crisis, which was focused mostly on three west African countries and killed more than 10,000 people. Some critics have said the WHO\u2019s slow response played a major role in allowing the epidemic to balloon. Zika is related to yellow fever and dengue. An estimated 80% of people that have it have no symptoms, making it difficult for pregnant women to know whether they have been infected. Anne Schuchat, principal deputy director of the Centers for Disease Control and Prevention, said: \u201cWe really do expect there to be a lot more travel-associated cases. For the average American who is not travelling this is not something they need to worry about. For people who are pregnant and considering travel to the affected areas, please take this seriously.\u201d She said living conditions in the US, including lower human density in urban areas and access to air conditioning, meant people were at less of a risk of contracting the virus than those living in South and Central American cities. \u201cWe don\u2019t have local transmission of the virus in the US right now,\u201d added Fauci. \u201cThere\u2019s essentially no risk at all because we don\u2019t have locally transmitted Zika virus in the US.\u201d \u201cWe believe this is a time-limited infection in women, men and children,\u201d said Schuchat. \u201cPeople have symptoms for up to about a week, not months and years of chronic viral infection. We know four out of five people with this infection have no symptoms.\u201d She added the Zika virus passed very quickly through the bloodstream and in most cases the virus would clear from the bloodstream within about a week. Asked when the babies of pregnant mothers could become infected with the virus, Schuchat said the foetus was most at risk of contracting microcephaly through Zika in the first trimester of the pregnancy. Scientists did not yet have \u201csufficient knowledge to know what effects in the second and third trimester\u201d, she cautioned. There has been one reported case of the Zika virus through \u201cpossible sexual transmission\u201d, while a second case was found in a man\u2019s semen. However, Schuchat highlighted that scientific research clearly showed Zika was \u201cprimarily transmitted through the bite of an infected mosquito\u201d."))
# train_data_y.append(['zikavirus'])
# train_data_x.append(html.unescape("Spraying pesticides will fail to deal with the Zika virus, a leading Kenyan entomologist has said this week. Dr Dino Martins spoke to the Guardian from his home in Laikipia about the virus that has been declared a public health emergency by the World Health Organisation. While pesticides are useful for removing flying adult mosquitoes that transmit the virus, he argues they will fail to deal with the epidemic that threatens to become a global pandemic. The virus was first detected in Uganda in 1947, although its exact origins are unknown. Martins makes an exception for indoor spraying. When pesticide is applied to walls where mosquitoes rest after feeding, it can be highly effective. But spraying in landscapes is extremely dangerous he warns: \u201cIt is a quick fix but you pay for it. You kill other species that would have predated on the mosquitoes. You also create a mosaic of sprayed and unsprayed low densities of chemicals that fosters the rapid evolution of resistance.\u201d Mosquitoes have life cycles of a week or less, and each generation is an opportunity for random mutations to occur that might predispose a group of mosquitoes to be immune to pesticides. \u201cIn addition, when you use chemicals, you are actually applying a selection pressure on mosquito populations that will drive them to become resistant,\u201d says Martins, who studied for his PhD at Harvard. \u201cWe are basically fighting an arms race with mosquitoes rather than cleverly understanding its life cycle and solving the problem there. Resistance can never evolve to getting rid of the breeding sites. But resistance will always evolve to the use of pesticides,\u201d says Martins, who runs Mpala Research Centre, a field station affiliated with Princeton, Smithsonian Institution, Kenya Wildlife Service and National Museums of Kenya. The explosion of mosquitoes in urban areas has driven the Zika crisis, and this has at least two causes, according to Martins. One is the lack of natural diversity to keep them under control and the other is lack of waste disposal and the proliferation of plastic. While photos this week show public areas being fumigated, Martins says that it is impossible to fumigate every corner of a habitat where mosquitoes might possibly breed. \u201cIt might seem easier to just to spray but pesticides will not work long term,\u201d he says. \u201cWe need to ask \u2013 what is the weakest point in the life cycle of this vector? For me, it is the larvae because they are fixed and findable. You can destroy them right there. Once the mosquitoes fly, it is far harder ... We need more investment in mosquito control at early rather than late stages.\u201d Martins has, alongside paleontologist Richard Leakey, successfully controlled malaria in part of Turkana, a county of extreme aridity. \u201cWe knew that mosquitoes breed in standing water, yet there was almost none around. It took us sometime to work out that they were breeding in the traditional shallow wells, which women dig and then leave once the quality of water has declined,\u201d says Martins. Treatment of active cases and indoor residual spraying using WHO-approved long-acting chemicals, combined with encouraging women and young people to fill in the wells, helped to end a severe localised epidemic. And yet even there Martins found that if people threw plastic bags into the well it increased the survival of mosquito larvae. He speculates that the ban on plastic bags in Rwanda under President Kagame might account for much of the decline in malaria in Rwanda, combined with improved case management and indoor spraying. \u201cWith mosquitoes, and this applies to Zika, chemicals are just one tool. We\u2019ve not invested in the broad research that will address this problem sustainably or in plastic recycling. Plastic persists and the bags have become a ubiquitous feature of life across east Africa,\u201d says Martins. Global warming is unhelpful too: mosquito species that typically occurred in lower altitude warmer areas are now able to survive in high altitude areas, leading to malaria and other diseases. Reflecting on the way the health and science community have dealt with Zika reveals important lessons for the future of global health. \u201cThis is where good surveillance, archiving of data, and public commons scientific spaces are important. It is highly unlikely that the blood samples from when it was discovered have survived - Uganda has gone through problems and science has not been well supported. Yet this is a public health crisis. Having historical information and supporting scientists in developing countries now will solve a lot of problems in future.\u201d Cathy Watson is chief of programme development at the World Agroforestry Centre in Nairobi, which addresses the links between landscapes and human health. Join our community of development professionals and humanitarians. Follow @GuardianGDP on Twitter."))
# train_data_x.append("Zika virus")
# tunisiaattack2015
# train_data_y.append(['tunisiaattack2015'])
# train_data_x.append(html.unescape("One story dominates the front pages, and many inside pages, of today\u2019s national newspapers: the murderous attack in Tunisia. With 30 Britons among the dead, that is to be expected. Amid the emotion and the fear, there is fact: 30 Brits are dead (Daily Mirror); Tunisia attack: police on alert amid fears UK toll will hit 30 (Guardian); \u201cAnd still the death toll climbs\u201d (i); Terror police on alert amid fears of UK attack (Times); and David Cameron: now the fightback begins (the Daily Telegraph\u2019s report on an article written by the prime minister for the paper). There is speculation: Tunisian killer may not have acted alone (Independent). There is defiance, exemplified by a couple\u2019s decision to get engaged after fleeing the guman: Our love is stronger than their hate (Sun); To hell with terror.. let\u2019s get married (Star) and \u201cOur love proves terror won\u2019t prevail\u201d (Metro). There are demands: Tell us if loved ones are dead or alive (Daily Mail) and Send in SAS to crush Jihadis (Daily Express). So what do the newspapers have to say in their editorials about the massacre in Sousse as they strive to come to terms with its ramifications? For the Telegraph, the major lesson of the incident is the need for Muslims in Britain, especially their religious leaders, to denounce the \u201cpoisonous ideology\u201d that gave rise to the atrocity \u201cat every turn \u2013 in schools, in the home and in the mosques.\u201d It concludes: \u201cDr Muhammad Tahir-ul-Qadri, a Pakistani theologian, recently issued a fatwa against suicide bombers, stating that they should be ostracised, not lauded as martyrs. He has suggested that British Muslims hold a mass march for peace to protest against the terrorists and everything they stand for. Many Muslims may argue that they should not need to demonstrate their disgust for these jihadi killers. But a resolute, systematic and outspoken denunciation of the theological rationale they use to justify their actions is always welcome.\u201d The Sun takes a similar line in a leading article headlined Muslim families must call cops. \u201cThe authorities have done amazing work stopping IS from succeeding here so far,\u201d it says, \u201cbut it\u2019s an impossible task expecting them to be able to shield us forever.\u201d It continues: \u201cWhether it\u2019s through social media, Islamist forums or direct contact with hate preachers, IS\u2019s poisonous ideology is inspiring too many British Muslims. Home secretary Theresa May is right to demand that Muslim families report their children to the police if they fear that they\u2019re becoming radicalised. It sounds brutal. But if they don\u2019t, they risk becoming the parents not of oddballs intrigued by radicalisation, but of cold-blooded terrorists.\u201d The Sun backs Cameron\u2019s argument that Muslim communities must do more to prevent Isis\u2019s poison from spreading. If not, \u201ctheir numbers are going to grow and grow.\u201d A similar view is evident in the op-ed article by Leo McKinstry in the Express, We must stand up to the extremism taking hold here. He writes: \u201cContrary to the fashionable talk about \u2018the vast majority\u2019 of moderates, 40% of Muslims in Britain want to see sharia law formally established here while 30% of Muslim students on British university campuses desire a caliphate and think that killing in the name of Islam is justified. Far from taking the fight to extremism our political class has allowed it to flourish. The vital work of our security forces has been undermined by human rights legislation and by anxiety about accusations of so-called Islamophobia.\u201d The Times takes a much wider view by considering the implications of the fragility of the democratic movements in north Africa. In its leader, Cherish Tunisia, it points to the \u201cdesperately forlorn\u201d hopes that once thrived in that country and elsewhere. It says: \u201cIf democracy fails or the economy craters in Tunisia, all that will remain of the Arab Spring will be war, autocracy and the obscenity of the so-called caliphate. The only significant difference between north Africa before and after its experiment with plural government will be that the region is now an even more lethal incubator of extremism than it was.\u201d But the paper regards Tunisia as \u201ca beachhead for civilisation in a region where civilisation is on the run.\u201d Yet Tunisia \u201cexports more fighters to Syria than any other country. Youth unemployment is at 35% and rising. Reforms to root out corruption and liberalise the labour market have stalled.\u201d For the Times, \u201cTunisia is a symbol of what is possible. Its ruling party is secular. Its leading Islamist party is avowedly democratic, resisting the idea of Islam as a source and limit of law.\u201d The paper concludes: \u201cIt is a reason to stand by Tunis come what may. Its brave experiment with democracy is too important to fail.\u201d The Independent is much more pessimistic about Tunisia\u2019s future because it believes people will shy away from choosing it as a holiday destination. In fact, says the paper, it is a further example of the way in which \u201cthe world is being closed off to casual visitors from our part of the world.\u201d It lists others: Afghanistan (once \u201can important stopping-off point on a \u2018hippie trail\u2019\u201d), Lebanon, Iran, Pakistan, Syria, Libya and Egypt. \u201cMass tourism has a gross and exploitative side to it,\u201d says the Indy, \u201cbut it is an important economic motor in many poor, developing countries.\u201d And it argues: \u201cOur world is effectively shrinking and, as our physical horizons are reduced, it is hard not to believe that our mental horizons will not suffer the same fate.\u201d Both the Guardian and the Mirror praise the reaction of the Tunisian people, including hotel staff, who helped holidaymakers during the attack. Staff and medics who ran towards the bullets showed \u201cextraordinary bravery\u201d, says the Guardian. Tunisia\u2019s prosperity is now threatened, it says, \u201cbecause of the shadow one madman has cast across its reputation as a holiday paradise\u201d and it concludes: \u201cThe terrorists\u2019 version of Islam is a twisted distortion. Real Islam stresses hospitality. Tunisians have shown what that looks like when it is fortified with courage.\u201d The Mirror agrees. \u201cIn the midst of the despair, we also discover the very best in people. The heroes, British and Tunisian, who risked their lives to save others... And the Tunisians now demonstrating against the Islamists.\u201d It says: \u201cConfronting the bloodthirsty fascists of the so-called Islamic State requires bravery so we should cheer the Tunisians making a stand. So as we mourn, let us welcome a glimmer of hope \u2013 supporting those in north Africa and the Middle East, most of them Muslims, in the frontline against Islamist wickedness.\u201d The Mail sounds a very different note by attacking the British government for its response to the incident. Why was the Foreign Office \u201cso slow in confirming the identities of the Islamist gunman\u2019s victims?\u201d It also launches an assault on \u201csocial media... for providing a vehicle for the vile Islamic State propaganda that fuelled Tunisian gunman Seifeddine Rezgui and the countless others like him who the security services fear are plotting similar \u2018lone wolf\u2019 attacks in Britain.\u201d Like the Telegraph and the Sun, it sees virtue in the plea by home secretary Theresa May to Muslim families to inform on their own children or friends: \u201cWhile the overwhelming majority of British Muslims abhor the terrorists \u2013 some in the Islamic community can do more to condemn and root out extremism.\u201d"))
# train_data_x.append("Tunisia attack 2015")
# munichshooting
# train_data_y.append(['munichshooting'])
# train_data_x.append(html.unescape("A teenager with German-Iranian citizenship has shot and killed nine people and wounded more than 15 at a shopping centre in Munich, in the third attack on civilians in Europe in eight days. The 18-year-old man, who police believe acted alone, is understood to have lived in Munich for up to two years. He reportedly shouted \u201cI am German\u201d during the prolonged attack on Friday evening, at the end of which he killed himself. Germany\u2019s third largest city was forced into lockdown after the gunman opened fire on diners in a McDonald\u2019s restaurant before moving to a nearby shopping mall. His motive was \u201ccompletely unclear\u201d, said Munich police chief Hubertus Andrae. There was no immediate evidence of an Islamist or other terrorist motive. Police raided the attacker\u2019s home in the early hours of Saturday, according to local media, but there were no details of his identity and he was not known to police. A video posted on Twitter appeared to show the gunman in a furious exchange with a bystander as the attack was going on. In the footage, an unseen man can be heard shouting abuse at a man who appears to be the attacker pacing the top of a car park. The unseen man can be heard telling other people with him that the man in the carpark has a gun, to which the man purported to be shooter responds: \u201cFucking Turks!\u201d The unseen man shouts: \u201cHe has loaded his gun. Get the cops here,\u201d to which the other man shouts back: \u201cI am German.\u201d The German chancellor, Angela Merkel, was due to meet her chief of staff, interior minister, and a host of intelligence officials on Saturday morning to review the incident, which comes in the wake of the Bastille Day truck atrocity in Nice and an axe attack in southern Germany. The French president, Fran\u00e7ois Hollande said the Munich shooting was a \u201cdisgusting terrorist attack\u201d aimed at stirring up fear across Europe. \u201cThe terrorist attack that struck Munich killing many people is a disgusting act that aims to foment fear in Germany after other European countries,\u201d Hollande said. \u201cGermany will resist, it can count on France\u2019s friendship and cooperation,\u201d he said, adding that he would speak to Merkel on Saturday morning. US intelligence officials, speaking to Reuters on condition of anonymity, said initial reports from their German counterparts indicated no apparent link between the shooter and Islamic State or other militant groups. A 15-year-old girl was among the dead, and at least 16 people, including several children, were in hospital; three were in critical condition, Andrae said. The gunman\u2019s body was found in a side street close to the mall. Police used a bomb disposal robot to search the site for explosives and booby traps, a local reporter said. Police stopped trains, buses and trams, closed highways to private cars and ordered citizens to stay in their homes as they searched for suspected killers, as false rumours of fresh attacks sent panic through the city. The transport network was reopened following the all-clear. The violence began just before 6pm, when the gunman, wearing a red backpack, opened fire at the McDonald\u2019s restaurant outside the Olympia shopping centre, near the site of the 1972 Olympic Games. Video apparently shot outside the restaurant showed people fleeing in terror as a gunman with a pistol surveyed the street then calmly and indiscriminately opened fire as terrified bystanders raced for cover. Emergency services raced to the site within minutes, but the gunman had apparently vanished. Police, fearing several attackers, searched the city, and a painstaking operation was launched to secure the shopping mall where dozens of shoppers and workers were still thought to be hiding. For several hours, as rumours about the number and location of attackers swept through Munich, and officers went slowly from store to store, there was a desperate vigil for loved ones trapped inside. \u201cMy 23-year-old daughter was part of a group that locked themselves inside H&amp;M to protect themselves. I spoke to her over the phone and she was crying, but then her battery ran out,\u201d said one father, weeping himself. He asked not to be named because of fears for his daughter. Asked if this was a terror attack, a police spokesman said: \u201cIf a man with a gun in a shopping centre opens fire and eight people are dead, we have to work on the assumption that this was not a normal crime and was a terrorist act.\u201d Merkel will convene a meeting of her security council, made up of senior ministers, on Saturday. Cansu Muyan, who lives near the Olympia shopping centre, said she had been inside the mall with her sister when the attack began. \u201cI suddenly saw everyone running past. Then a shopkeeper told us all to leave as quickly as possible so we all started running as well,\u201d she said. Other witnesses reported hearing shots inside and outside the shopping mall, known locally as the OEZ. \u201cI was shopping when I heard three shots, then we ran out and about 40 seconds later we heard five shots from outside,\u201d said Florian Horn, 33. Staff in the mall were still in hiding more than an hour after the attack, an employee told Reuters by telephone. \u201cMany shots were fired. I can\u2019t say how many, but it\u2019s been a lot,\u201d said the employee, who declined to be identified. \u201cAll the people from outside came streaming into the store and I only saw one person on the ground who was so severely injured that he definitely didn\u2019t survive.\u201d \u201cI ran out, I was so afraid, and then some people brought me and several others into their garden and apartment where we found safety,\u201d said Jennifer Hartel, who had lost a shoe fleeing the attack and was still shaking three hours after the violence. The horror of the bloodshed was followed by hours of uncertainty, as police raced to track down the gunman captured on video and up to two other reported attackers. Police used a smartphone warning system, Katwarn, to urge people to stay at home, and used social media to ask locals and journalists not to share photos or video of police action to avoid helping any suspects on the run. Residents responded by sharing pictures of pets and other cuddly animals under hashtags also used for news of the attack, and offering those stranded in the city a place to stay. Hospitals were on emergency alert with staff, including doctors, surgeons and nurses, called in to await casualties. Among the parts of the city to be evacuated was Munich central station. People were reported to have screamed and scrambled over railway platforms as the police ordered them to leave the station. Germany\u2019s elite unit, its SAS equivalent GSG9, was flown in to support local security forces. Armed and masked but dressed otherwise unassumingly in T-shirts, trainers and shorts they were spotted in the vicinity of the local police who were on the scene within minutes of the first emergency call having been received. It is the second attack in Bavaria in less than a week. Security forces have been on high alert after a teenage refugee attacked train passengers near the city of W\u00fcrzburg with an axe and a knife. Islamic State claimed responsibility for the train attack, but authorities have said the attacker was likely to have acted alone. Flags will fly at half mast on official buildings across Germany on Saturday. The country\u2019s interior minister, Thomas de Maizi\u00e8re \u2013 currently flying back from New York, will head straight to Munich on Saturday morning . The UK\u2019s Foreign Office issued an alert warning British citizens in Munich to follow the instructions of the authorities. Speaking at the UN in New York, Boris Johnson, the foreign secretary, said: \u201cEverybody is shocked and saddened by what has taken place. Our thoughts are very much with the victims, their families, with the people of Munich.\u201d \u201cIf, as seems very likely, this is another terrorist incident, then I think it proves once again that we have a global phenomenon now and a global sickness that we have to tackle both at source \u2013 in the areas where the cancer is being incubated in the Middle East \u2013 and also of course around the world.\u201d"))
# train_data_x.append("Munich shooting")


##### Generate vectors for words in training data Articles    #####
print("Generating frequency vectors for the words present in the training set. This will take a while...")
#
vectorizer = TfidfVectorizer(min_df=0.005, max_df=0.90, stop_words='english',  smooth_idf=True,
                             norm="l2", sublinear_tf=False, use_idf=True, ngram_range=(1, 3))
X = vectorizer.fit_transform(train_data_x)

##### Generate vectors for words in test data Articles    #####
xTest = vectorizer.transform(test_data_x)

##### Level binarizer for Class variable of Train set     #####
print("Converting the class variable to binary matrix...")
mlb_train = MultiLabelBinarizer(classes = labels)
print(mlb_train)
train_data_y = mlb_train.fit_transform(train_data_y)
print(train_data_y)
print(len(train_data_y[1]))

##### Fitting a One Vs Rest Classifier using the Naive Bayes classifier       #####
print("Training the model...")
classifier = OneVsRestClassifier(SGDClassifier(alpha=0.0001, learning_rate="optimal",
                                               class_weight="balanced", n_iter=100, n_jobs=-1)).fit(X, train_data_y)
classifier
##### Getting the Predictions for labels for test data        #####
print("Getting the predictions for labels on Test Data...")
y_pred = classifier.predict(xTest)

print(y_pred)

count = 0
for i in y_pred:
    for j in i:
        if j == 1:
            count += 1
print(count)

pred_df = pd.DataFrame(y_pred)
pred_df.to_csv("prediction.csv", header=labels, index=False)

print(set(labels).difference(unique_train_labels))

print("Time taken: " + str(datetime.datetime.now() - start_time))