class subgroup_rulesets(object):
    def __init__(self, X, ITE, print_message=False):
        self.df = X
        self.Y = deepcopy(ITE)
        self.print_message=print_message

    def generate_rules(self, threshold, maxlen, N, method = 'randomForest'):
        self.maxlen = maxlen
        self.threshold = threshold 
        self.Ytilde = (self.Y > self.threshold).astype(int)
        neg_df = 1-self.df #df has negative associations
        neg_df.columns = [name.strip() + '_neg' for name in self.df.columns]
        df = pd.concat([self.df,neg_df], axis = 1)
        if method =='fpgrowth' and maxlen<=3:
            itemMatrix = [[item for item in df.columns if row[item] ==1] for i,row in df.iterrows() ]
            pindex = np.where(self.Y==1)[0]
            nindex = np.where(self.Y!=1)[0]
            if self.print_message:
                print('Generating rules using fpgrowth')
            start_time = time.time()
            rules= fpgrowth([itemMatrix[i] for i in pindex],supp = supp,zmin = 1,zmax = maxlen)
            rules = [tuple(np.sort(rule[0])) for rule in rules]
            rules = list(set(rules))
            start_time = time.time()
            if self.print_message:
                print('\tTook %0.3fs to generate %d rules' % (time.time() - start_time, len(rules)))
        else:
            rules = []
            start_time = time.time()
            for length in range(1,maxlen+1,1):
                n_estimators = min(pow(df.shape[1],length),4000)
                clf = RandomForestClassifier(n_estimators = n_estimators,max_depth = length)
                #clf = RandomForestRegressor(n_estimators = n_estimators,max_depth = length)
                clf.fit(self.df,self.Ytilde)
                for n in range(n_estimators):
                    rules.extend(extract_rules(clf.estimators_[n],df.columns))
            rules = sorted([list(x) for x in set(tuple(x) for x in rules)])
            if self.print_message:
                print('\tTook %0.3fs to generate %d rules' % (time.time() - start_time, len(rules)))
        self.screen_rules(rules,df,N) # select the top N rules using secondary criteria, information gain
        self.cond_names = self.df.columns.append(neg_df.columns)
        
    def screen_rules(self,rules,df,N):
        if self.print_message:
            print('Screening rules using information gain')
        start_time = time.time()
        itemInd = {}
        for i,name in enumerate(df.columns):
            itemInd[name] = i
        indices = np.array(list(itertools.chain.from_iterable([[itemInd[x] for x in rule] for rule in rules])))
        len_rules = [len(rule) for rule in rules]
        indptr =list(accumulate(len_rules))
        indptr.insert(0,0)
        indptr = np.array(indptr)
        data = np.ones(len(indices))
        ruleMatrix = csc_matrix((data,indices,indptr),shape = (len(df.columns),len(rules)))
        mat = np.matrix(df) * ruleMatrix
        lenMatrix = np.matrix([len_rules for i in range(df.shape[0])])        
        Z = (mat ==lenMatrix).astype(int)
        CATEs = np.array([(self.Y * Z) / np.sum(Z, axis=0)]).flatten()
        select = np.where(CATEs > self.threshold)[0]
        self.rules = [rules[i] for i in select]
        self.RMatrix = np.array(Z[:,select])
        if self.print_message:
            print('\tTook %0.3fs to generate %d rules' % (time.time() - start_time, len(self.RMatrix[0])))
    
    def set_parameters(self, alpha, maxcomplex=999):
        self.max_group_size = len(self.df)
        self.max_effect_size = np.max(self.Y)
        min_effect_size = np.min(self.Y)
        self.Y -= min_effect_size
        self.threshold -= min_effect_size
        self.ATE = np.mean(self.Y)
        self.alpha = alpha
        self.maxcomplex = maxcomplex
    
    # @param fg_scale scaling factor for the logarithmic function determing probability neighborhood is fine grained
    def find_soln(self, Niteration = 5000, Nchain = 3, q = 0.1, fg_scale=5, fg_switch=.5, eta=0.95, init = []):
        all_obfn_vals = defaultdict(list)
        all_acpt_probs = defaultdict(list)
        all_deltas = defaultdict(list)
        # print 'Searching for an optimal solution...'

        nRules = len(self.rules)
        self.rules_len = [len(rule) for rule in self.rules]
        maps = defaultdict(list)
        T0 = 1000
        split = 0.7*Niteration
        
        for chain in range(Nchain):
            # initialize with a random pattern set
            if init !=[]:
                rules_curr = init[:]
            else:
                N = sample(range(1,min(8,int(np.floor(self.maxcomplex/self.maxlen))),1),1)[0]
                rules_curr = sample(range(nRules),N)
                while self.maxcomplex - sum([self.rules_len[x] for x in rules_curr]) < 0 :  # repeat until not too complex
                    rules_curr = sample(range(nRules),N)
            rules_curr_norm = self.normalize(rules_curr)
            pt_curr = float('-inf') 
            # rules_curr is rule ind for RMatrix
            maps[chain].append([-1,pt_curr,rules_curr,[self.rules[i] for i in rules_curr]])
            

            # calibrate cooling schedule
            cur_sol, cur_sol_norm = deepcopy(rules_curr), deepcopy(rules_curr_norm)
            cal_deltas = []
            for i in range(100):
                fg=random() < 0.5
                new_sol, new_sol_norm = self.propose(cur_sol[:], cur_sol_norm[:],q,fg)
                delta = self.compute_objfn(new_sol) - self.compute_objfn(cur_sol)
                cal_deltas.append(abs(delta)) 
                cur_sol, cur_sol_norm = new_sol[:],new_sol_norm[:]
            T0 = -np.mean(cal_deltas)/np.log(.8)#np.percentile(cal_deltas, 75)#
            
            for iter in range(Niteration):
                if iter>=split:
                    p = np.array(range(1+len(maps[chain])))
                    p = np.array(list(accumulate(p)))
                    p = p/p[-1]
                    index = find_lt(p,random())
                    rules_curr = maps[chain][index][2][:]
                    rules_curr_norm = maps[chain][index][2][:]
                #rules_new, rules_norm = self.propose(rules_curr[:], rules_curr_norm[:],q)
                
                # choose neighborhood
                fg = (random() <= 1/(1+np.exp(-fg_scale*(iter-(Niteration*fg_switch))/Niteration))) # logistic scaling for probability of fine grained neighborhood
                # propose new rule set 
                rules_new, rules_norm = self.propose(rules_curr[:], rules_curr_norm[:],q,fg)#max(0.1, 1 - iter / (0.2 * Niteration)),fg)
                loss =  self.compute_objfn(rules_new)
                
                #T = T0**(1 - iter/Niteration)
                pt_new = loss
                delta = pt_new - pt_curr
                all_deltas[chain].append(delta)
                if False:#if iter < Niteration*.1 and np.isfinite(delta):
                    all_deltas[chain].append(delta)
                    if len(all_deltas[chain]) > 10:
                        T0 = np.percentile(np.abs(all_deltas[chain]), 75)
                    #print("delta stats:", np.min(deltas), np.max(deltas), np.percentile(deltas, [25, 50, 75]))
                T = T0*(eta ** iter) #T0 ** (1-iter/Niteration) ## # # #
                alpha = np.exp(float(delta)/T)  # acceptance probability
#                print(f"new:{pt_new}, curr:{pt_curr}, alpha:{alpha}")
                all_acpt_probs[chain].append(min(1, alpha))
                
                if pt_new > maps[chain][-1][1]:
                    maps[chain].append([iter,loss,rules_new,[self.rules[i] for i in rules_new]])
                    if self.print_message:
                        print('\n** chain = {}, max at iter = {} ** \n loss = {}, rule = {}'.format(chain, iter, loss, rules_new))
                if random() <= alpha:
                    rules_curr_norm,rules_curr,pt_curr = rules_norm[:],rules_new[:],pt_new
                    all_obfn_vals[chain].append(loss)
        pt_max = [maps[chain][-1][1] for chain in range(Nchain)]
        index = pt_max.index(max(pt_max))
        # print '\tTook %0.3fs to generate an optimal rule set' % (time.time() - start_time)
        final_soln = maps[index][-1][3]
        
        return final_soln, all_obfn_vals, all_acpt_probs, all_deltas
        
    def propose(self, rules_curr,rules_norm,q,fg):
        nRules = len(self.rules)
        len_rs = len(rules_curr)  # length of current ruleset
        complex_budget = self.maxcomplex - sum([self.rules_len[x] for x in rules_curr]) # remaining amount possible to increase complexity
#        fg=random() < 0.5
#        fg=False
        if fg:
            #######
            start = time.perf_counter()
            #######
            rs_expanded = [deepcopy(self.rules[i]) for i in rules_curr]  # rs as a [] of [conditions]
            if complex_budget == 0:  # if at max complexity, don't choose a rule with length 1 (can't cut or add condition)
                index=sample([x for x in range(len_rs) if len(rs_expanded[x])>1],1)[0]
            else:
                index = sample(range(len_rs),1)[0] # pick rule
            # pick action
            if len(rs_expanded[index])==1:
                move = sample([['add'], ['cut', 'add']],1)[0]
            elif len(rs_expanded[index])==self.maxlen or complex_budget==0:
                move = sample([['cut'], ['cut', 'add']],1)[0]
            else:
                move = sample([['add'], ['cut'], ['cut', 'add']],1)[0]

            replace = len(move) > 1
                
            if move[0]=='cut':
                if random() < q: # randomly choose a condition
                    try:
                        cut_cond = sample(rs_expanded[index],1)[0]
                        rs_expanded[index].remove(cut_cond)
                    except:
                        print(rs_expanded)
                        print(index)
                else: 
                    neighbors = {self.compute_objfn(rules_curr): deepcopy(rs_expanded)}
                    for cond in rs_expanded[index]:
                        dupe = deepcopy(rs_expanded)
                        dupe[index].remove(cond)
                        try:
                            neighbors[self.compute_objfn([self.rules.index(x) for x in dupe])] = dupe
                        except:
                            continue
                    rules_curr = [self.rules.index(x) for x in neighbors[max(neighbors.keys())]]
                rules_norm = self.normalize(rules_curr)
                complex_budget = self.maxcomplex - sum([self.rules_len[x] for x in rules_curr])
                rs_expanded = [deepcopy(self.rules[i]) for i in rules_curr]  # rs as a [] of [conditions]
                move.remove('cut')
            
            if min([len(x) for x in self.rules])==0:
                print(replace)
                print(rules_curr)
                print(rs_expanded)
                print([i for i, val in enumerate([len(x) for x in self.rules]) if val==0])
            
            l0 = min([len(x) for x in rs_expanded])==0
            if l0:
                print(replace)
                print(rules_curr)
                print(rs_expanded)
                print([self.rules[i] for i in rules_curr])
                print(index)
            
            if len(rs_expanded[index]) < self.maxlen and len(move) > 0 and move[0]=='add':
                candidates = [cond for cond in self.cond_names if cond not in rs_expanded[index]]
                if l0:
                    print(candidates)
                    
                #if False:   
                if random()<q: # random condition
                    for cond in sample(candidates, len(candidates)):
                        dupe = deepcopy(rs_expanded)
                        try: 
                            dupe[index].append(cond)
                            rules_curr = [self.rules.index(x) for x in dupe]  # check is valid rule
                            #rs_expanded = dupe
                            break
                        except:
                            continue
                else:
                    neighbors = {self.compute_objfn(rules_curr): deepcopy(rs_expanded)}
                    for cond in candidates:
                        dupe = deepcopy(rs_expanded)
                        dupe[index].append(cond)
                        if l0:
                            print(f"rules_curr:{rules_curr}")
                            print(f"dupe: {dupe}")
                        try:
                            neighbors[self.compute_objfn([self.rules.index(x) for x in dupe])] = dupe
                        except:
                            continue
                    if l0:
                        print(neighbors)
                    rules_curr = [self.rules.index(x) for x in neighbors[max(neighbors.keys())]]
                rules_norm = self.normalize(rules_curr)
                
                
                ###
            end = time.perf_counter()

            self.total_time_a += (end - start)
            self.count_a += 1
                ###

                
        else: 
            
            ######
            start = time.perf_counter()
            ######
            
            covered = (np.sum(self.RMatrix[:,rules_curr],axis = 1)>0).astype(int)
            higher_uncovered = np.where((self.Ytilde==1) & [1-x for x in covered])[0] # uncovered examples with effect > threshold

            # randomly sample an action
            eg_h = None
            if len(higher_uncovered)==0:
                p_add = 0
            else: 
                p_add = 1  # possible to add rule
                eg_h = sample(list(higher_uncovered),1)[0]
            
            if len_rs==1: # only one rule
                if p_add > 0: 
                    if random() < 0.5:
                        move = ['add']
                    else: 
                        move = ['cut', 'add'] # replace
                else: # no possible rules to add
                    move = ['clean']
            elif p_add ==0 or complex_budget < self.maxlen: # at least two rules AND (no rules to add OR adding rule would make too complex, and 
                if random() < 0.5:
                    move = ['cut']
                else: 
                    move = ['cut', 'add'] # replace
            else: # no constraints on actions
                move = sample([['add'], ['cut'], ['add', 'cut']],1)[0]
                
        
            if move[0]=='cut':
                """ cut """
                candidates = rules_curr 
                if random()<q:  # cut random rule
                    cut_rule = sample(candidates,1)[0]
                    rules_curr.remove(cut_rule)
                else:  # cut rule that increases objective the most
                    neighbors = {}
                    for candidate in candidates:
                        dupe = deepcopy(rules_curr)
                        dupe.remove(candidate)
                        neighbors[self.compute_objfn(dupe)] = dupe
                    rules_curr = neighbors[max(neighbors.keys())] 
                rules_norm = self.normalize(rules_curr)
                complex_budget = self.maxcomplex - sum([self.rules_len[x] for x in rules_curr])
                move.remove('cut')

            if len(move)>0 and move[0]=='add':
                """ add """
#                added_rule = None
                if eg_h is None:  # for when replace & p_add=0
                    covered = (np.sum(self.RMatrix[:,rules_curr],axis = 1)>0).astype(int)
                    higher_uncovered = np.where((self.Ytilde==1) & [1-x for x in covered])[0]
                    if len(higher_uncovered) > 0:
                        eg_h = sample(list(higher_uncovered),1)[0]
                    else:
                        return rules_curr, rules_norm # no examples above threshold uncovered
                candidates_initial = list(set(np.where(self.RMatrix[eg_h, :] == 1)[0]) - set(rules_curr))
                candidates = [x for x in candidates_initial if self.rules_len[x] <= complex_budget]  # get candidates for rules to add satisfying complexity constraint
                if random()<q:  # pick random rule
                    added_rule = sample(range(nRules),1)[0]
                    rules_curr.append(added_rule)
                else:  # pick rule that maximizes gain in objective fn
                    neighbors = {}
                    for candidate in candidates:
                        dupe = deepcopy(rules_curr)
                        dupe.append(candidate)
                        neighbors[self.compute_objfn(dupe)] = dupe
                    rules_curr = neighbors[max(neighbors.keys())]  
                    rules_norm = self.normalize(rules_curr)

                if len(move)>0 and move[0]=='clean':
                    remove = []
                    for i,rule in enumerate(rules_norm):
                        Yhat = (np.sum(self.RMatrix[:,[rule for j,rule in enumerate(rules_norm) if (j!=i and j not in remove)]],axis = 1)>0).astype(int)
                        TP,FP,TN,FN = getConfusion(Yhat,self.Y)
                        if TP+FP==0:
                            remove.append(i)
                    for x in remove:
                        if (x in rules_norm):
                            rules_norm.remove(x)
                    return rules_curr, rules_norm
                
                ######
                end = time.perf_counter()

                self.total_time_b += (end - start)
                self.count_b += 1
                ######
                
        return rules_curr, rules_norm

    def compute_objfn(self,rules):
        covered = (np.sum(self.RMatrix[:,rules],axis = 1)>0).astype(int)
        CATE = np.mean(self.Y[covered > 0])
        loss = ((np.sum(covered)/self.max_group_size)**self.alpha) * ((CATE/self.max_effect_size))
        return np.log(loss)
    
    def normalize(self, rules_new):
        try:
            rules_len = [len(self.rules[index]) for index in rules_new]
            rules = [rules_new[i] for i in np.argsort(rules_len)[::-1][:len(rules_len)]]
            p1 = 0
            while p1<len(rules):
                for p2 in range(p1+1,len(rules),1):
                    if set(self.rules[rules[p2]]).issubset(set(self.rules[rules[p1]])):
                        rules.remove(rules[p1])
                        p1 -= 1
                        break
                p1 += 1
            return rules[:]
        except:
            return rules_new[:]
    
    