#http://www.philchen.com/2015/08/08/how-to-make-a-scalable-python-web-app-using-flask-and-gunicorn-nginx-on-ubuntu-14-04

#I know that this is the ugliest API ever written, I would fix it if I had more time

#to install on ec2 Ubuntu
#sudo /bin/dd if=/dev/zero of=/var/swap.1 bs=1M count=1024
#sudo /sbin/mkswap /var/swap.1
#sudo /sbin/swapon /var/swap.1
#sudo apt-get install -y libatlas-base-dev gfortran python-dev build-essential g++
#sudo pip install numpy
#sudo pip install scipy
#sudo pip install sklearn
#sudo pip install cloudpickle
#sudo pip install pandas
#sudo swapoff /var/swap.1
#sudo rm /var/swap.1


from flask import Flask, Response, jsonify, make_response, abort, request
import cloudpickle, string
from numpy import array, nonzero
from pandas import to_numeric

app = Flask(__name__)

forum_vectorizer=cloudpickle.load(open("forum_vectorizer.pkl", "rb" ) )
forum_vecrepresent=cloudpickle.load(open("forum_vecrepresent.pkl", "rb" ) )
forum_classifier=cloudpickle.load(open("forum_classifier.pkl", "rb" ) )
forum_db=cloudpickle.load(open("forum_db.pkl", "rb" ) )
# forum_mention=cloudpickle.load(open("doctortext_labels.pkl", "rb" ) )
disease_db=cloudpickle.load(open("disease_db.pkl", "rb" ) )
cond_sim=cloudpickle.load(open("disease_similarity.pkl", "rb" ) )
cond_names = [x['name'] for x in disease_db]

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

def find_similar_experiences(text,filt,count):
    # words = text.encode("utf8").lower().translate(string.maketrans(string.punctuation, ' '*len(string.punctuation))).split()
    # vec = np.array(forum_vectorizer.infer_vector(words))
    vec = forum_vectorizer.transform([text])
    filt_t = [(filt in l) for l in forum_db['label']]
    filt_i = nonzero(filt_t)[0]
    temp_db = forum_db.loc[filt_t,:]
    temp_vecr = forum_vecrepresent[filt_i,:]
    sim = temp_vecr.dot(vec.transpose())
    temp_db['sims']=to_numeric(array(sim.todense()).flatten())
    # sorty =%time temp_db.sort_values('sims',ascending=False)[['post','askertxt']]
    truncy =temp_db.nlargest(count,columns=['sims'])
    return truncy[['post','askertxt']].T.to_dict().values()
    # return forum_db.loc[filtit,:].sort_values('sims',ascending=False).head(count)[['post','askertxt']].T.to_dict().values()

def find_possible_conditions(text,count):
    # words = text.encode("utf8").lower().translate(string.maketrans(string.punctuation, ' '*len(string.punctuation))).split()
    # vec = np.array(forum_vectorizer.infer_vector(words))
    vec = forum_vectorizer.transform([text])
    predict_p = forum_classifier.predict_proba(vec)
    prob = array([a[:,1] for a in predict_p]).T.tolist()[0]
    mysub=[(disease_db[i]['name'],p) for i,p in enumerate(prob) if (disease_db[i]['name']!='cancer') and (disease_db[i]['name']!='anxiety')]
    return sorted(mysub,key=lambda x: -x[1])[:count]

    # prob_condition = [(disease_db[i]['name'],a) for i,a in enumerate(prob_condition)]
    # return sorted(prob_condition,key=lambda x: -x[1])[:count]

def get_clinical_occurence(condition):
    lookup=[(i,d['clin_freq'],d['clin_ref']) for i,d in enumerate(sorted(disease_db,key=lambda x: -x['clin_freq'])) if d['name']==condition]
    if(len(lookup)==0):
        return None
    if(lookup[0][1]==0):
        return None
    return {'clin_ref':lookup[0][2], \
        'clin_text': "There were <strong>" +str(int(lookup[0][1])) +" cases</strong> of "+condition+" reported last year.  It is ranked as the " +str(int(1+lookup[0][0]))+"th most common disease reported by the CDC."}

def get_wiki_occurence(condition):
    lookup=[(i,d['wiki_freq'],d['wiki_ref']) for i,d in enumerate(sorted(disease_db,key=lambda x: -x['wiki_freq'])) if d['name']==condition]
    if(len(lookup)==0):
        return None
    if(lookup[0][1]==0):
        return None
    return {'wiki_ref':lookup[0][2], \
        'wiki_text': "People viewed the article for " +condition+" at least <strong>"+str(int(lookup[0][1])) +" times</strong> on wikipedia in the last month.  That makes it more popular than <strong>" \
        +str(int(100*(len(disease_db)-lookup[0][0])/len(disease_db)))+"% </strong> of all conditions."}

def get_forum_occurence(condition):
    lookup=[(i,d['forum_freq']) for i,d in enumerate(sorted(disease_db,key=lambda x: -x['forum_freq'])) if d['name']==condition]
    if(len(lookup)==0):
        return None
    if(lookup[0][1]==0):
        return None
    return {'forum_text': "There are <strong>" +str(int(lookup[0][1])) +" forum posts</strong> about "+condition+" in our dataset.  That makes it the " +str(int(1+lookup[0][0]))+"th most commonly discussed topic on the health forums we curate."}


def generate_similarity_csv(conditions,probs,count):
    yield 'source,target,sim\n'
    percond = count/len(conditions)
    subset=zip(probs,conditions)
    connect = 1
    for j,c in enumerate(conditions):
        yield '%s,%s,%s' % (c, c, probs[j]) +'\n'
        sim = cond_sim[[i for i,x in enumerate(cond_names) if x == c],:][0]
        s = sorted(zip(sim,cond_names), key=lambda x: x[0], reverse=True)
        if(s[percond][0]<connect):
            connect=s[percond][0]
        subset.extend([(probs[j],b) for (a,b) in s[:percond]])
    if(connect<0.1):
        connect=0.1
    seen = set()
    ans = []
    for item in subset:
        if item[1] not in seen:
            ans.append(item)
            seen.add(item[1])
    subprob=[a for (a,b) in ans]
    subcond=[b for (a,b) in ans]
    inds=[i for i,x in enumerate(cond_names) if x in subcond]
    subsim=cond_sim[:,inds][inds,:]
    for i in range(subsim.shape[0]):
        for j in range(i+1,subsim.shape[0]):
            if(subsim[i,j]>=connect*0.9):
                yield '%s,%s,%s' % (subcond[i], subcond[j], (float(subprob[i])+float(subprob[j]))/2) +'\n'







# http://127.0.0.1:5000/linkhealth/api/v1.0/statistics/chlamydia
@app.route('/linkhealth/api/v1.0/statistics/<condition>', methods=['GET'])
def get_statistics(condition):
    clinic=get_clinical_occurence(condition)
    wiki=get_wiki_occurence(condition)
    forum=get_forum_occurence(condition)
    return jsonify({'clinic':clinic,'wiki':wiki,'forum':forum})






# http://127.0.0.1:5000/linkhealth/api/v1.0/similarity/chlamydia;syphillis/0.2;0.6
@app.route('/linkhealth/api/v1.0/similarity/<conditions>/<probs>', methods=['GET'])
def get_similarity_default(conditions,probs):
    return get_similarity(conditions,probs,10)

@app.route('/linkhealth/api/v1.0/similarity/<conditions>/<probs>/<count>', methods=['GET'])
def get_similarity(conditions,probs,count):
    return Response(generate_similarity_csv(conditions.split(';'),probs.split(';'),count), mimetype='text/csv')









#http://127.0.0.1:5000/linkhealth/api/v1.0/experiences/%22I%20have%20some%20red%20rash%20spots%20on%20my%20thighs%22/condition=scabies
@app.route('/linkhealth/api/v1.0/experiences/<text>', methods=['GET'])
def get_experiences_default(text):
    return get_experiences(text,"")

@app.route('/linkhealth/api/v1.0/experiences/<text>/<condition>', methods=['GET'])
def get_experiences(text,condition):
    sim_exp = find_similar_experiences(text,condition,10)
    return jsonify({'sim_exp': sim_exp})






#http://127.0.0.1:5000/linkhealth/api/v1.0/conditions/"I have some red rash spots on my thighs"
@app.route('/linkhealth/api/v1.0/conditions/<text>', methods=['GET'])
def get_condition_default(text):
    return get_conditions(text,5)

@app.route('/linkhealth/api/v1.0/conditions/<text>/<count>', methods=['GET'])
def get_conditions(text,count):
    # return jsonify({'answer':find_possible_conditions(text,int(count))})
    ret = find_possible_conditions(text,int(count))
    return jsonify({'conditions':[a for (a,b) in ret],'probs':[b for (a,b) in ret]})



if __name__ == '__main__':
    app.run(debug=True)
