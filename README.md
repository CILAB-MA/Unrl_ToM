# Dip_ToM
## TODO
### Environment

### Dipblue Agent
- [ ] Sensitive response of reject nego(for Messy ver)
- [ ] If counterpart attack me, reflect in relations(Messy ver)
- [ ] Check Retreat Phase of Order Handler
### Data Preprocssing
Data preprocessing procedure are 3 stages. `log_collector.py`, `one_epi`, and `Make Whole Data`.
 
 - `target_dst.npy` : `shape : (num_data,  1)` - where src go
 
 - `target_order.npy` : `shape : (num_data, 1)` - what src do
 
 - `me_index.npy` : `shape : (num_data, num_past + 1)` - index of power of me
 
 - `other_index.npy` : `shape : (num_data, num_past + 1)` - index of power of other
 
 - `past_internal.npy` : `shape : (num_data, num_past, num_step, 6)`
    - `peaces[other], wars[other], trust_ratio[other] of me` : 3
    - `peaces[me], wars[me], trust_ratio[me] of other` : 3
    - TODO -> now it is not like this, plz just use last 3 dim
 
 - `past_order.npy` : `shape : (num_data, num_past, num_step, 181)`
    - `order type [H, -, S, C]` : 4
    - `srcs location` : 81
    - `dsts location` : 81
    - `src_power` : 7
    - `dst_power` : 7
 
 - `past_map_tactician.npy` : `shape : (num_data, num_past, num_step, 120)`
    - `Scalar map values of all regions (unit_type, loc)` : 120
    - They are calculated from map_tactician adviser
    
 
 - `curr_internal.npy` : `shape : (num_data, curr_step, 6)`
    - same as past
    - TODO -> now it is not like this, plz just use last 3 dim
 
 - `curr_map_tactician.npy` : `shape : (num_data, curr_step, 120)`
    - same as past
 
 - `curr_src.npy` : `shape : (num_data, 81)`
 
 - `me_weights.npy` : `shape : (num_data, 4)`
    - TODO -> now it is not like this
    
 - `other_weights.npy` : `shape : (num_data, 4)`
    - TODO -> now it is not like this
 


## Codes
### dipbluebot
DipBlue bot, DAIDE engine.  
An implementation of "[Dipblue: A diplomacy agent with strategic and trust reasoning](https://paginas.fe.up.pt/~niadr/PUBLICATIONS/2015/ICAART_2015_77_CR.pdf)"  
We refer to [Ferreira, André's code](https://github.com/andreferreirav2/dipblue).

### collect data
```
python utils/data_collector_new.py -na 7 -e 1000 -ev 100 -np 5 -ns 200 -n 0 --base_dir /data/cilab_ma/data/dip_tom --num_cpu 40
```
    --num_agent, -na, : The number of players(dipblue) in a game.
    --num_episode, -e : The number of train episode produced from a single cpu.
    --num_eval, -ev :  The number of validation episode produced from a single cpu.
    --number, -n : A run id for identification
    --num_past, -np : The number of past transition for training in an episode. 
    --num_step, -ns : The number of step 
    --alpha, -a : 
    --base_dir, -b : A directory for saved data.
    --curr_step, -c : 
    --data_type, -w : 
    --num_cpu, -p : The number of cpu to collect data


### run
```python test.py```
