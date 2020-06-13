# -*- coding: utf-8 -*-

from flask import render_template,request, session
import sqlite3
from gensim.models.doc2vec import Doc2Vec
from app import server


# app = Flask(__name__)

w = []
model1=None

model1 = Doc2Vec.load('./model/model')

    
@server.route('/')
def tags():
    items_on_page = server.config['ITEMS_PER_PAGE']
    db = sqlite3.connect('./data/db.sqlite')
    cur = db.cursor()
    cur.execute('select * from tags')
    results = cur.fetchall()
    length = len(results)
    last = int(len(results)//items_on_page)
    page = request.args.get('num')
    if not str(page).isnumeric():
        page=1
        page=int(page)
    if int(page)==1:
        n = '/?num='+str(int(page)+1)
        p = "#"
    elif int(page)==last:
        p = '/?num='+str(int(page)-1)
        n = "#"
    else:
        p = '/?num='+str(int(page)-1)
        n = '/?num='+str(int(page)+1)

    results = results[(int(page)-1)*items_on_page:int(page)*items_on_page]
    print(last)

    cur.close()
    return render_template('tags.html',tags=results,next=n,prev=p,length=length,page=int(page))

@server.route('/articles', methods=['GET', 'POST'])
def articles():
    items_on_page = server.config['ITEMS_PER_PAGE']
    #r = str()
    r = [request.args.get('type',type=str)]
    name = r[0].replace('+',' ')
    db = sqlite3.connect('./data/db.sqlite')
    c=db.cursor()
    c.execute('select id,title,Topic1,Author1,Author2 from data where Topic1 = ?',(name,))
    results = c.fetchall()
    length=len(results)
    last = int(len(results)//items_on_page)
    page = request.args.get('num')
    name = name.replace(' ','+')
    if not str(page).isnumeric():
        page=1
    if int(page)==1:
        n = '/articles?type='+name+'&num='+str(int(page)+1)
        p = "#"
    elif int(page)==last:
        p = '/articles?type='+name+'&num='+str(int(page)-1)
        n = "#"
    else:
        p = '/articles?type='+name+'&num='+str(int(page)-1)
        n = '/articles?type='+name+'&num='+str(int(page)+1)
    results = results[(int(page)-1)*items_on_page:int(page)*items_on_page]
    c.close()
    return render_template('articles.html',message=results,title=str(request.args.get('type')),next=n,prev=p,length=length,page=int(page),name=str(name))


@server.route('/models',methods=['GET','POST'])
def models():
    # global w
    global model1
    similarity = None
    results = None
    # model = Doc2Vec.load('/home/deviantpadam/Downloads/model')
    w = str(request.args.get('type'))
        # return res
    # print(w)
    a = []
    similarity = model1.docvecs.most_similar(w)
    # print(similarity)
    simdi = {}
    for i,j in similarity:
        simdi.setdefault(i,[]).append(j)
    db = sqlite3.connect('./data/db.sqlite')
    c=db.cursor()
    c.execute('SELECT * FROM data where id = ?',(w,))
    topic_open = c.fetchall()
    for i in similarity:
            a.append(i[0])
    sql="select * from data where id in ({seq})".format(seq=','.join(['?']*len(a)))

    c.execute(sql, a)
    results = c.fetchall()
    c.close()
    return render_template('similar.html', message=simdi,
                           results=results,topic_open=topic_open)

@server.route('/about')
def about():
    return render_template('about.html',title='About')

@server.route('/search',methods=['GET','POST'])
def search():
    db = sqlite3.connect('./data/db.sqlite')
    c = db.cursor()
    items_on_page = server.config['ITEMS_PER_PAGE']
    if request.method == 'POST':
        search = request.form['search']
        session['name'] = search
    if 'name' in session:
        c.execute('SELECT * FROM data WHERE data =?',(session['name'],))
        results = c.fetchall()
    # print(topic_open[0])
    length=len(results)
    last = int(len(results)//items_on_page)
    page = request.args.get('num')
    if not str(page).isnumeric():
        page=1
    if int(page)==1:
        n = '/search?num='+str(int(page)+1)
        p = "#"
    elif int(page)==last:
        p = '/search?num='+str(int(page)-1)
        n = "#"
    else:
        p = '/search?num='+str(int(page)-1)
        n = '/search?num='+str(int(page)+1)
    results = results[(int(page)-1)*items_on_page:int(page)*items_on_page]
    return render_template('search.html',title =session['name'],message = results,next=n,prev=p,length=length,page=int(page))

    
@server.errorhandler(500)
def internal_error(error):

    return render_template('home.html')

