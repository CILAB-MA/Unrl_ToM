from environment.utils import *

class Order:

    def __init__(self):
        self.orders = ['H', '-', 'S', 'C']
        self.unit_type = ['A', 'F', None]

    def parse(self, order, owners):
        word = order.split()
        unit_type, unit_loc, order_type = word[:3]
        src_infos = None
        dst_infos = None
        # move unit
        if order_type == '-':
            dst = word[-1]
            if dst not in owners:
                dst_power = None
            else:
                dst_power = owners[dst]
            dst_infos = [dst, dst_power]
        # support for hold unit
        elif (order_type == 'S') and ('-' not in word):
            src = word[-1]
            if src not in owners:
                src_power = None
            else:
                src_power = owners[src]
            src_infos = [src, src_power]
            dst_infos = [src, src_power]
        # support and convoy for unit
        elif (order_type in ['S', 'C']) and ('-' in word):
            src = word[word.index('-') - 1]
            dst = word[-1]
            if src not in owners:
                src_power = None
            else:
                src_power = owners[src]
            if dst not in owners:
                dst_power = None
            else:
                dst_power = owners[dst]
            dst_infos = [dst, dst_power]
            src_infos = [src, src_power]

        return unit_type, unit_loc, order_type, src_infos, dst_infos

    def parse_dipgame(self, str_order, loc2power):

        word = str_order.split()

        if len(word) != 1:
            unit_type, unit_loc, order_type = word[:3]
            src_infos = None
            dst_infos = None
        else:
            unit_type = None
            unit_loc = None
            order_type = None
            src_infos = None
            dst_infos = None


        if order_type == "H":
            order_type = "HLD"

        elif (order_type == '-') and ('VIA' in word):
            dst = word[-2]
            dst_power = loc2power[dst]
            dst_infos = [dst, dst_power]
            order_type = "CVY_MOVE"

        # {unit_type} {unit_loc} - D
        elif order_type == "-" and ('D' in word):
            order_type = "DSB"

        elif order_type == "D":
            order_type = "DSB"

        # elif order_type == "-" and ('D' in word):
        #     order_type = "REM"

        elif order_type == '-':
            dst = word[-1]
            dst_power = loc2power[dst]
            dst_infos = [dst, dst_power]
            order_type = "MTO"

        elif (order_type == 'S') and ('-' not in word):
            src = word[-1]
            src_power = loc2power[src]
            src_infos = [src, src_power]
            order_type = "SUP"

        elif (order_type in ['S', 'C']) and ('-' in word):
            src = word[word.index('-') - 1]
            src_power = loc2power[src]
            dst = word[-1]
            dst_power = loc2power[dst]
            dst_infos = [dst, dst_power]
            src_infos = [src, src_power]
            if order_type == 'S':
                order_type = "SUPMTO"
            else:
                order_type = "CVY"

        # {unit_type} {unit_loc} - {dst}
        elif order_type == "R":
            dst = word[-1]
            dst_power = loc2power[dst]
            dst_infos = [dst, dst_power]
            order_type = "RTO"

        elif order_type == "B":
            order_type = "BLD"

        elif "WAIVE" in word:
            order_type = "WVE"


        return unit_type, unit_loc, order_type, src_infos, dst_infos

    ####### Movement Phase Orders #######
    def hold(self, unit_type, unit_loc):
        return unit_type + ' ' + unit_loc + ' H'

    def move(self, unit_type, unit_loc, dst):
        return '{} {} - {}'.format(unit_type, unit_loc, dst)

    def support(self, unit_type, unit_loc, src_loc, src_type):
        return '{} {} S {} {}'.format(unit_type, unit_loc, src_type, src_loc)

    def support_move(self, unit_type, unit_loc, src_type, src_loc, dst):
        return '{} {} S {} {} - {}'.format(unit_type, unit_loc, src_type, src_loc, dst)

    def convoy(self, unit_type, unit_loc, src_type, src_loc, dst):
        return '{} {} C {} {} - {}'.format(unit_type, unit_loc, src_type, src_loc, dst)

    def convoy_move(self, unit_type, unit_loc, dst):
        return '{} {} - {} VIA'.format(unit_type, unit_loc, dst)

    ####### Retreat Phase  #######
    def retreat(self, unit_type, unit_loc, dst):
        return '{} {} R {}'.format(unit_type, unit_loc, dst)

    def disband(self, unit_type, unit_loc):
        # in NPD, remove == disband
        return '{} {} D'.format(unit_type, unit_loc)

    ####### Build Phase  #######
    def build(self, unit_type, unit_loc):
        return '{} {} B'.format(unit_type, unit_loc)

    def waive(self):
        return 'WAIVE'
