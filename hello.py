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
import numpy as np

app = Flask(__name__)

forum_vectorizer=cloudpickle.load(open("forum_vectorizer.pkl", "rb" ) )
forum_vecrepresent=cloudpickle.load(open("forum_vecrepresent.pkl", "rb" ) )
forum_classifier=cloudpickle.load(open("forum_classifier.pkl", "rb" ) )
forum_db=cloudpickle.load(open("forum_db.pkl", "rb" ) )
disease_db=cloudpickle.load(open("disease_db.pkl", "rb" ) )
disease_sim=cloudpickle.load(open("disease_similarity.pkl", "rb" ) )

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

def find_similar_experiences(text,filters):
    # vec = forum_vectorizer.transform([text])
    words = text.encode("utf8").lower().translate(string.maketrans(string.punctuation, ' '*len(string.punctuation))).split()
    vec = np.array([forum_vectorizer.infer_vector(words)])
    sim = forum_vecrepresent.dot(vec.transpose())
    forum_db['sims'] = sim
    toret = []
    for it in forum_db.sort_values('sims',ascending=False).itertuples():
        if (isinstance(it[3],str) and any(f in it[3] for f in filters)):
            toret.append({'post':it[1],'response':it[2],'label':it[3],'sim':it[4]})
        if(it[4]==0):
            break
    return toret

def find_possible_conditions(text,count):
    # vec = forum_vectorizer.transform([text])
    words = text.encode("utf8").lower().translate(string.maketrans(string.punctuation, ' '*len(string.punctuation))).split()
    vec = np.array([forum_vectorizer.infer_vector(words)])
    predict_prob = forum_classifier.predict_proba(vec)
    prob_condition=predict_prob.tolist()[0]
    return sorted([(disease_db[i]['name'],p) for i,p in enumerate(prob_condition)],key=lambda x: -x[1])[:count]


def get_clinical_occurence(condition):
    lookup=[(i,d['clin_freq']) for i,d in enumerate(sorted(disease_db,key=lambda x: -x['clin_freq'])) if d['name']==condition]
    if(len(lookup)==0):
        return "Sorry, we don't have clinical stats on "+condition+" yet. Check back later."
    return "There were " +str(int(lookup[0][1])) +" cases of "+condition+" reported last year.  It is ranked as the " +str(int(1+lookup[0][0]))+"th most common infectious disease reported by the CDC."



def generate_similarity_csv(conditions,probs,count):
    percond = count/len(conditions)
    subset=zip(probs,conditions)
    connect = 1
    for j,c in enumerate(conditions):
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
    yield 'source,target,sim\n'
    for i in range(subsim.shape[0]):
        for j in range(i+1,subsim.shape[0]):
            if(subsim[i,j]>=connect*0.9):
                yield '%s,%s,%s' % (subcond[i], subcond[j], (float(subprob[i])+float(subprob[j]))/2) +'\n'


#http://ec2-54-208-15-210.compute-1.amazonaws.com/linkhealth/api/v1.0/statistics/chlamydia
@app.route('/linkhealth/api/v1.0/statistics/<condition>', methods=['GET'])
def get_statistics(condition):
    clinic=get_clinical_occurence(condition)
    wiki='No wikipdeia info yet'
    forum='No forum data yet.'
    return jsonify({'clinic':clinic, 'wiki':wikie, 'forum': forum})


# explain yourself
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

@app.route('/linkhealth/api/v1.0/experiences/<text>/<conditions>', methods=['GET'])
def get_experiences(text,conditions):
    sim_exp = find_similar_experiences(text,conditions.split(';'))
    return jsonify({'sim_exp': sim_exp})


#http://127.0.0.1:5000/linkhealth/api/v1.0/conditions/"I have some red rash spots on my thighs"
@app.route('/linkhealth/api/v1.0/conditions/<text>', methods=['GET'])
def get_condition_default(text):
    return get_conditions(text,5)

@app.route('/linkhealth/api/v1.0/conditions/<text>/<count>', methods=['GET'])
def get_conditions(text,count):
    ret = find_possible_conditions(text,count)
    return jsonify({'conditions':[a for (a,b) in ret],'probs':[b for (a,b) in ret]})



if __name__ == '__main__':
    app.run(debug=True)
