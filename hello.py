#http://www.philchen.com/2015/08/08/how-to-make-a-scalable-python-web-app-using-flask-and-gunicorn-nginx-on-ubuntu-14-04

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
import cloudpickle
import numpy as np

app = Flask(__name__)

import nltk, string

#can also try porterstemmer or lancasterstemmer
stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

from sklearn.feature_extraction.text import TfidfVectorizer

def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))


forum_vectorizer=cloudpickle.load(open("forum_vectorizer.pkl", "rb" ) )
forum_classifier=cloudpickle.load(open("forum_classifier.pkl", "rb" ) )
forum_tfidf=cloudpickle.load(open("forum_tfidf.pkl", "rb" ) )
forum_db=cloudpickle.load(open("forum_db.pkl", "rb" ) )
cond_names=cloudpickle.load(open("condition_names.pkl", "rb" ) )
cond_sim=cloudpickle.load(open("condition_similarity.pkl", "rb" ) )
cond_stat=cloudpickle.load(open("condition_statistics.pkl", "rb" ) )

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

def find_similar_experiences(text,filters):
    vec = forum_vectorizer.transform([text])
    sim = forum_tfidf.dot(vec.transpose())
    forum_db['sims']=sim.todense()
    toret=[]
    for it in forum_db.sort_values('sims',ascending=False).itertuples():
        if (not filters) or (isinstance(it[3],str) and any(f in it[3] for f in filters)):
            toret.append({'post':it[1],'response':it[2],'label':it[3],'sim':it[4]})
        if(it[4]==0):
            break
    return toret

def find_possible_conditions(text,count):
    vec = forum_vectorizer.transform([text])
    prob = forum_classifier.predict_proba(vec)
    parr=np.array([a[:,1] for a in prob])[:,0].tolist()
    return [a for (a,b) in sorted(zip(cond_names,parr), key=lambda x: x[1], reverse=True)[:count]]

def get_clinical_occurence(condition):
    try:
        num=cond_stat.loc[cond_stat[0]==condition,'us_freq'].values[0]
        rank=cond_stat['us_freq'].rank(ascending=False)[cond_stat[0]==condition].values[0]
        if(rank>41):
            rank=-1
        return {'occurences':num,'rank':rank}
    except:
        return {'broken':'sorry broken'}



def generate_similarity_csv(conditions,count):
    percond = count/len(conditions)
    subcond=list(conditions)
    connect = 1
    for c in conditions:
        sim = cond_sim[[i for i,x in enumerate(cond_names) if x == c],:][0]
        s = sorted(zip(sim,cond_names), key=lambda x: x[0], reverse=True)
        if(s[percond][0]<connect):
            connect=s[percond][0]
        subcond.extend([b for (a,b) in s[:percond]])
    if(connect<0.1):
        connect=0.1
    subcond=list(set(subcond))
    inds=[i for i,x in enumerate(cond_names) if x in subcond]
    subsim=cond_sim[:,inds][inds,:]
    yield 'source,target,sim'
    for i in range(subsim.shape[0]):
        for j in range(i+1,subsim.shape[0]):
            if(subsim[i,j]>=connect):
                yield '%s,%s,%s' % (subcond[i], subcond[j], subsim[i,j]) +'\n'


#not implemented yet
@app.route('/linkhealth/api/v1.0/statistics/<condition>', methods=['GET'])
def get_statistics(condition):
    clinic=get_clinical_occurence(condition)
    return jsonify({'clinic':clinic})

#not implemented yet
@app.route('/linkhealth/api/v1.0/similarity/<conditions>', methods=['GET'])
def get_similarity(conditions):
    count = request.args.get('count')
    if(not count):
        count = 10
    return Response(generate_similarity_csv(conditions.split(';'),count), mimetype='text/csv')

#http://127.0.0.1:5000/linkhealth/api/v1.0/experiences/%22I%20have%20some%20red%20rash%20spots%20on%20my%20thighs%22?condition=scabies
@app.route('/linkhealth/api/v1.0/experiences/<text>', methods=['GET'])
def get_experiences(text):
    condition = request.args.get('condition')
    if(condition):
        sim_exp = find_similar_experiences(text,condition.split(';'))
    else:
        sim_exp = find_similar_experiences(text,[])
    return jsonify({'sim_exp': sim_exp})

#http://127.0.0.1:5000/linkhealth/api/v1.0/conditions/"I have some red rash spots on my thighs"
@app.route('/linkhealth/api/v1.0/conditions/<text>', methods=['GET'])
def get_conditions(text):
    count = request.args.get('count')
    if(not count):
        count = 5
    conditions = find_possible_conditions(text,count)
    return jsonify({'conditions':conditions})


if __name__ == '__main__':
    app.run(debug=True)
