
import sys
import os,sys,string,re
from math import sqrt,log
import operator

_debug=0
_K=5
_R=2

from ExpertSystem import Paper
from ExpertSystem import Expert
from ExpertSystem import ExpertSystem
import io

# Function that Return an Array of all Expert with their attributed Paper array
def get_expert_papers(expert_list_file, expert_dir):
    reviewers_dict = {}

    author_format = expert_dir + "/{}.txt"
    with open(expert_list_file, 'r') as file:
        expert_names = file.readlines()

        for author in expert_names:
            # creating expert objects and adding then to dictionary
            author = (author.strip())
            with io.open(author_format.format(author), 'r',
                         encoding='utf-8', errors='ignore') as infile, \
                io.open('temp.txt', 'w', encoding='utf-8', errors='ignore') \
                    as outfile:
                for line in infile:
                    print(*line.split(), file=outfile)

            file1 = open("temp.txt", "r")
            for eachline in file1.readlines():
                p1 = Paper(eachline, author, -1, True)
                reviewer = reviewers_dict.get(author)
                if(reviewer is not None):
                    reviewer.papers.append(p1)
                else:
                    e1 = Expert(p1, author)
                    reviewers_dict[author] = e1

    print(len(reviewers_dict))
    return reviewers_dict


# Returns an array of Papers for which Experts are to be allocated
def get_paper_candidates(title_file, abstract_file):
        research_papers = []

        # now for papers for which we neeed recommendation
        file = open(title_file, "r")
        lines = file.readlines()

        with io.open(abstract_file, 'r', encoding='utf-8',
                     errors='ignore') as infile, \
                io.open('temp.txt', 'w', encoding='utf-8', errors='ignore') \
                as outfile:

            for line in infile:
                print(*line.split(), file=outfile)

        file1 = open("temp.txt", "r")
        contents = file1.readlines()
        # list of paper objects for which we need recommendation
        research_papers = []

        # adding to research paper list
        for line, content in zip(lines, contents):
            line = line.strip()
            content = str(line) + str(content)
            p1 = Paper(content, -1, -1, False)
            research_papers.append(p1)

        print(len(research_papers))
        return research_papers


# assigning unique index to Each paper
def assign_paper_uid(expertlist, research_papers):
    i = 0
    # assigning unique index to each paper
    all_paper = []
    all_paper_objects = []
    for expert in expertlist:
        for paper in expert.papers:
            paper.assignindex(i)
            all_paper.append(paper.contents)
            all_paper_objects.append(paper)
            i = i + 1

    for paper in research_papers:
        paper.assignindex(i)
        all_paper.append(paper.contents)
        all_paper_objects.append(paper)
        i = i + 1

    return (all_paper, all_paper_objects)


# Outputs the results to the file
def save_to_file(output_file, recommend):
    with open(output_file, 'w') as output:
        for paper in recommend:
            for expert, score in paper:
                output.write("{}:{},".format(expert, score))

            output.write("\n")


def main():
    method = 2
    output_file = 'output.txt'

    # a dictionary of Expert object indexed by names
    print("Loading Expert Data..")
    reviewers_dict = get_expert_papers('evaluation/short_listed_revs.txt',
                                       'title_abstract_data')
    print("Loading Paper Candidates")
    research_papers = get_paper_candidates("evaluation/titles.txt",
                                           'evaluation/abstracts.txt')
    expertlist = list(reviewers_dict.values())
    (all_paper, all_paper_objects) = assign_paper_uid(expertlist,
                                                      research_papers)
    print("Generating the TF-IDF Matrix..")
    ES = ExpertSystem(n_clusters=5, experts_per_paper=30)
    tfidf_mat = ES.get_tfidf(all_paper)

    # generating similarity matrix
    print("Generating the similarity matrix..")
    similarity = ES.get_cosine_sim(tfidf_mat)
    with open('validate.txt', 'w') as validation:
        print("Finding Optimal K")
        for i in range(3, 12):
            recommend = []
            ES.n_clusters = i
            print("K value : {}".format(i))
            labels = ES.get_kmeans(tfidf_mat)
            recommend = ES.algorithm2(labels, similarity, expertlist,
    	                                  research_papers)
            save_to_file(output_file, recommend)
            pres = calc_prec(output_file)
            print ('\nPrecision: '+str(pres)+'\n')
            validation.write("{},{}".format(i, pres))
    print("Done!")


#....................................................
def calc_prec(infile):
	Revs=get_revs()
	ofname=infile
	ifname='evaluation/short_listed_papers.txt'

	res=open(ofname,'r').read()
	res_list=res.split('\n')

	gfname='evaluation/gold_standard.txt'
	ih=open(gfname,'r')

	i=-1;N=-1;Prec=0;Chk=0;M=0

	#read gold-standard
	for line in ih:
		i=i+1
		ref_record=line.strip()
		if ref_record != 'UNK':
			N=N+1
			res_record=res_list[N].strip()
			(prec,chk,flag)=process_paper(Revs,ref_record,res_record,i)
			# print (str(i)+' '+str(N)+' '+str(prec))
			Prec=Prec+prec
			Chk=Chk+chk
			if flag:
				M=M+1
	ih.close()
	Prec=float(Prec)/float(M)
	Chk=Chk/float(M)
	print (str(Prec))

	return str(Prec)
	#print Chk

#....................................................
def process_paper(Revs,ref_record,res_record,pap_id):
	ref_parts=ref_record.split(',')
	ref={}
	flag=False
    # Processing gold_standard.txt
	for part in ref_parts:
		part=part.strip()
		if part != '':
			(rev,score)=part.split(':')
			ref[rev]=int(score)
			if ref[rev]>=_R:
				flag=True

    # Processing paper_reviewer_score.txt
	res_parts=res_record.split(',')
	res={}
	for part in res_parts:
		part=part.strip()
		if part != '':
			(rev,score)=part.split(':')
			if rev.find('_')==-1:
				revs=rev.split()
				revname=revs[0]+'_'+revs[1]
			else:
				revname=rev
			if revname in Revs:
				res[revname]=float(score)
	(prec,chk)=compute_prec(ref,res,pap_id)
	return(prec,chk,flag)


#....................................................
def compute_prec(ref,res,pap_id):
    sorted_revs = sorted(res.items(), key=operator.itemgetter(1), reverse=True)
    prec=0;drec='';flag=False
    for i in range(_K):
        record=sorted_revs[i]
        (name,score)=record
        drec=drec+name+':'+str(score)+','
        if name in ref:
            if ref[name]>=_R:
                prec=prec+1
                flag=True
    chk=0
    for rev in ref:
        if ref[rev]>=_R:
            chk=chk+1
    if chk>_K:
        chk=_K
    chk=chk/float(_K)
    prec=prec/float(_K)
    return(prec,chk)


#....................................................
def get_revs():
	ifname='evaluation/short_listed_revs.txt'
	shlst_revs=open(ifname,'r').read().strip().split('\n')
	return(shlst_revs)

#....................................................
def create_shlst_revs():
	gfname='evaluation/gold_standard.txt'
	ofname='evaluation/short_listed_revs.txt'
	ih=open(gfname,'r')
	oh=open(ofname,'w')
	Revs=[]
	for line in ih:
		line=line.strip()
		if line!='UNK':
			parts=line.split(',')
			for part in parts:
				part=part.strip()
				if part != '':
					#print part;#debug()
					(rev,score)=part.split(':')
					oh.write(rev+'\n')
					Revs.append(rev)
	ih.close()
	oh.close()
	return(Revs)

#....................................................
def debug():
    stop_str=raw_input("Press q to quit.")
    if stop_str is 'q':
        sys.exit(1)
#....................................................

if __name__=='__main__':
	main()
