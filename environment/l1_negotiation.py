
class Negotiation:

    def __init__(self, level=1):

        self.level = level
        self.nego_level = ['P', 'A', 'R', 'W']  # propose(send), accept(recv), reject(recv), withdraw
        self.nego_lang = ['Propose to ', 'Accept to', 'Reject to', 'Withdraw']
        self.deal_level = ['C', 'A'] # Commit, Agree
        self.deal_lang = ['Commit', 'Agree']
        self.content_level = ['P', 'AL', 'D', 'A', 'N']  # predicate, do, and, not
        self.content_lang = ['Peace', 'Alliance', 'Do', 'And', 'Not']

    def parse(self, message):
        word = message.split()
        phase, nego_l, sender, receiver, deal_l = word[:5]
        if deal_l == 'AGREE':
            agree_power = word[5]
            deals = [deal_l, agree_power]
            word = word[6:]
        else:
            commit_power, committed_power = word[5:7]
            deals = [deal_l, commit_power, committed_power]
            word = word[7:]
        cont_l = word[0]

        if cont_l  == 'PEACE':
            relation_power = word[1]
            conts = [cont_l, relation_power]

        elif cont_l == 'ALLIANCE':
            ally = word[1]
            enemy = word[2]
            conts = [cont_l, ally, enemy]

        elif cont_l == 'DO':

            do_ind = message.index('DO')
            order = message[do_ind + 3:]
            conts = [cont_l, order]
        # not for [and, not]
        return phase, nego_l, deals, conts, sender, receiver, message

    # -----TOP LEVEL MESSAGES-----
    def propose2str(self, sender, recipients, deal, phase):
        if type(recipients) == list:
            recipients = str(recipients)[1:-1].replace(' ', '').replace("'", '')
        return '{} PROPOSE {} {} {}'.format(phase, sender, recipients, deal)

    def accept2str(self, sender, recipients, deal, phase):
        if type(recipients) == list:
            recipients = str(recipients)[1:-1].replace(' ', '').replace("'", '')
        return '{} ACCEPT {} {} {}'.format(phase, sender, recipients, deal)

    def reject2str(self, sender, recipients, deal, phase):
        if type(recipients) == list:
            recipients = str(recipients)[1:-1].replace(' ', '').replace("'", '')
        return '{} REJECT {} {} {}'.format(phase, sender, recipients, deal)

    # -----SECOND LEVEL MESSAGES-----
    def commit2str(self, sender, recipients, offer):
        if type(recipients) == list:
            recipients = str(recipients)[1:-1].replace(' ','').replace("'",'')
        return 'COMMIT {} {} {}'.format(sender, recipients, offer)

    def agree2str(self, recipients, offer):

        if type(recipients) == list:
            recipients = str(recipients)[1:-1].replace(' ','').replace("'",'')
        return 'AGREE {} {}'.format(recipients, offer)

    # -----THIRD LEVEL MESSAGES-----
    def do2str(self, order):
        return 'DO {}'.format(order)

    def ally2str(self, ally, against):
        if type(against) == list:
            against = str(against)[1:-1].replace(' ','').replace("'",'')
        if type(ally) == list:
            ally = str(ally)[1:-1].replace(' ','').replace("'",'')

        return 'ALLIANCE {} {}'.format(ally, against)

    def peace2str(self, peace):
        if type(peace) == list:
            peace = str(peace)[1:-1].replace(' ', '').replace("'", '')
        return 'PEACE {}'.format(peace)


if __name__ == '__main__':
    nego = Negotiation()
    nego.parsing('A A ITA AL ITA ENG', 'RUS', 'ITA')
    nego.parsing('P C ITA RUS D MAR - PAR', 'RUS', 'ITA')

