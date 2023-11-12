import random
import numpy as np

class Env:
    def __init__(self):

        '''
        S = <leader_index, square_index, Hp, Mp>
        '''

        # env stuff 
        self.number_of_squares = 365
        self.current_square_index = 0

        # team stuff
        self.team = Team()
        self.leader = self.team.member_list[self.team.leader_index]
        self.P, self.D, self.T = self.leader.P, self.leader.D, self.leader.T

    def return_stage(self) -> int:
        '''
        Returns the current stage level [0, 4] based on current square index.
        '''

        return self.current_square_index // (self.number_of_squares // 3)     #  (curr) // (121) in [0, 3]
    
    def encounter(self) -> tuple:
        '''
        Game-like encounter
        '''
        stage = self.return_stage() 
        P, D, T = self.P, self.D, self.T
        
        # Calculate the weighted average based on stage - mean(weighted_avg) -> [0, 10]
        if stage == 0:
            weighted_avg = 0.9*P + 0.05*D + 0.05*T

        elif stage == 1:
            weighted_avg = 0.05*P + 0.9*D + 0.05*T

        elif stage == 2 or stage == 3:
            weighted_avg = 0.05*P + 0.05*D + 0.9*T

        # Normalize the weighted average to range [0, 1]
        normalized_avg = weighted_avg / 10
        
        # produce the distribution; # average the outcome for 5 trials; 
        sigma = 0.5
        dist_list = [np.clip(5 * np.random.normal(normalized_avg - 1.0, sigma), -6, 6) for _ in range(5)]
        prob_dist = sum(dist_list) / len(dist_list)

        # 0 is HP, 1 is MORALE
        hp_or_morale = random.choice([0, 1])   

        return (prob_dist, hp_or_morale)

    def reset(self, for_test=False):
        # env stuff
        self.number_of_squares = 365
        self.current_square_index = 0

        if not for_test:
            # team stuff
            self.team = Team()
            self.team.HP, self.team.Morale = 200, 200

        else:
            self.team = testTeam()
            self.team.HP, self.team.Morale = 200, 200
        
        self.team.leader_index = np.random.randint(0, 3)
        self.leader = self.team.member_list[self.team.leader_index]
        self.P, self.D, self.T = self.leader.P, self.leader.D, self.leader.T

        return (self.team.leader_index, self.current_square_index, self.team.HP, self.team.Morale)

    def step(self, action: int) -> tuple:
        '''
        The step function of the environment.
        Action is the "new" leader index.
        Action == -1 means no update.
        '''

        # change leader
        self.team.set_leader_to(action)
        self.leader = self.team.member_list[self.team.leader_index]
        self.P, self.D, self.T = self.leader.P, self.leader.D, self.leader.T

        # make 5 moves, calculate cumulative reward
        reward, Hp_or_Morale = self.encounter()

        # add reward to HP or MP
        if Hp_or_Morale: self.team.HP += reward
        else: self.team.Morale += reward

        # this is 1 step; update square index
        self.current_square_index += 1

        # determine transition components
        next_state = (self.team.leader_index, self.current_square_index, self.team.HP, self.team.Morale)
        done = True if self.team.HP <= 0 or self.team.Morale <= 0 or self.current_square_index >= 365 else False

        # if done: print(f"finished with HP: {self.team.HP} and MP: {self.team.Morale}")
        if done and self.current_square_index == 365: reward += 5

        return next_state, reward, done, {}

class Team:
    def __init__(self, number_of_people=3):

        # Stats
        self.HP = 100
        self.Morale = 100
 
        # TEAM-related
        self.number_of_people = number_of_people
        self.member_list = self.generate_team()

        # Leader
        self.leader_index = np.random.randint(0, 3)
        self.set_leader_to(self.leader_index)

    def generate_team(self):
        member_list = []
        for _ in range(self.number_of_people):
            new_member = Member()
            member_list.append(new_member)

        return member_list
    
    def set_leader_to(self, leader_index):

        # 기존 리더 비활성화 
        self.member_list[self.leader_index].is_leader = False

        # 리더 업데이트
        self.leader_index = leader_index
        self.member_list[self.leader_index].is_leader = True

class testTeam(Team):
    def __init__(self, number_of_people = 3):
        super().__init__(number_of_people)
        self.member_list = self.generate_team()
    
    def generate_team(self):
        member_list = []
        custom_list = [(9,1,1), (1,9,1), (1,1,9)]

        for i in range(self.number_of_people):
            new_member = Member(custom=custom_list[i])
            member_list.append(new_member)

        return member_list

class Member:
    def __init__(self, custom=False):
        '''
        P, D, T ranges from [0, 10)
        '''
        if custom is False:

            self.P = random.random() * 10
            self.D = random.random() * 10
            self.T = random.random() * 10

        else:
            self.P, self.D, self.T = custom

        self.is_leader = False

        # for logging
        # print()
        # print(f"Member Initialized with stats: {self.P, self.D, self.T}")
        # print()
