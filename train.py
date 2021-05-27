import tensorflow as tf
import datetime
import os
from config import config
from data_generator import DataGenerator
from model import Net
from visual import overlay, visualize_short_offsets
from post_proc import *

writer = tf.summary.create_file_writer(os.path.join(config.LOG_DIR, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
checkpoint_path = os.path.join(config.CHECKPOINT_PATH, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

def tf_repeat(tensor, repeats):
    """
    From  https://github.com/tensorflow/tensorflow/issues/8246
    
    Args:

    input: A Tensor. 1-D or higher.
    repeats: A list. Number of repeat for each dimension, length must be the same as the number of dimensions in input

    Returns:
    
    A Tensor. Has the same type as input. Has the shape of tensor.shape * repeats
    """
    expanded_tensor = tf.expand_dims(tensor, -1)
    multiples = [1] + repeats
    tiled_tensor = tf.tile(expanded_tensor, multiples = multiples)
    repeated_tensor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
    return repeated_tensor

def kp_map_loss(kp_maps_true,kp_maps_pred,unannotated_mask,crowd_mask):
    loss = tf.keras.backend.binary_crossentropy(kp_maps_true,kp_maps_pred)
    loss = loss*crowd_mask*unannotated_mask
    loss = tf.math.reduce_mean(loss)*config.LOSS_WEIGHTS['heatmap']
    return loss

def short_offset_loss(short_offset_true,short_offsets_pred,kp_maps_true):
    loss = tf.math.abs(short_offset_true-short_offsets_pred)/config.KP_RADIUS
    loss = loss*tf_repeat(kp_maps_true,[1,1,1,2])
    loss = tf.math.reduce_sum(loss) / (tf.math.reduce_sum(kp_maps_true)+1)
    return loss*config.LOSS_WEIGHTS['short']

def mid_offset_loss(mid_offset_true,mid_offset_pred,kp_maps_true):
    loss = tf.math.abs(mid_offset_pred-mid_offset_true)/config.KP_RADIUS
    recorded_maps = []
    for mid_idx, edge in enumerate(config.EDGES + [edge[::-1] for edge in config.EDGES]):
        from_kp = edge[0]
        recorded_maps.extend([kp_maps_true[:,:,:,from_kp], kp_maps_true[:,:,:,from_kp]])
    recorded_maps = tf.stack(recorded_maps,axis=-1)
    # print(recorded_maps)
    loss = loss*recorded_maps
    loss = tf.math.reduce_sum(loss)/(tf.math.reduce_sum(recorded_maps)+1)
    return loss*config.LOSS_WEIGHTS['mid']

def long_offset_loss(long_offset_true,long_offsets_pred,seg_true,crowd_mask,unannotated_mask,overlap_mask):
    loss = tf.math.abs(long_offsets_pred-long_offset_true)/config.KP_RADIUS
    instances = seg_true*crowd_mask*unannotated_mask*overlap_mask
    loss = loss*instances
    loss = tf.math.reduce_sum(loss)/(tf.math.reduce_sum(instances)+1)
    return loss*config.LOSS_WEIGHTS['long']

def segmentation_loss(seg_true,seg_pred,crowd_mask):
    loss = tf.keras.backend.binary_crossentropy(seg_true,seg_pred)
    loss = loss*crowd_mask
    return tf.math.reduce_mean(loss)*config.LOSS_WEIGHTS['seg']

def get_losses(ground_truth,outputs,step, mode="TRAIN"):
    kp_maps_true, short_offset_true, mid_offset_true, long_offset_true, seg_true, crowd_mask, unannotated_mask, overlap_mask = ground_truth
    kp_maps, short_offsets, mid_offsets, long_offsets, seg_mask = outputs
    losses = []
    kp_map_loss_value = kp_map_loss(kp_maps_true,kp_maps,unannotated_mask,crowd_mask)
    short_offset_loss_value = short_offset_loss(short_offset_true,short_offsets,kp_maps_true)
    mid_offset_loss_value = mid_offset_loss(mid_offset_true,mid_offsets,kp_maps_true)
    long_offset_loss_value = long_offset_loss(long_offset_true, long_offsets,seg_true,crowd_mask,unannotated_mask,overlap_mask)
    segmentation_loss_value = segmentation_loss(seg_true,seg_mask,crowd_mask)
    
    if step % config.LOG_SCALAR_INTERVAL is 0 and mode is "TRAIN":
        with writer.as_default():
            tf.summary.scalar("kp_map_loss", kp_map_loss_value, step=step)
            tf.summary.scalar("short_offset_loss", short_offset_loss_value, step=step)
            tf.summary.scalar("mid_offset_loss", mid_offset_loss_value, step=step)
            tf.summary.scalar("long_offset_loss", long_offset_loss_value, step=step)
            tf.summary.scalar("segmentation_loss", segmentation_loss_value, step=step)
            
    losses.append(kp_map_loss(kp_maps_true,kp_maps,unannotated_mask,crowd_mask))
    losses.append(short_offset_loss(short_offset_true,short_offsets,kp_maps_true))
    losses.append(mid_offset_loss(mid_offset_true,mid_offsets,kp_maps_true))
    losses.append(long_offset_loss(long_offset_true, long_offsets,seg_true,crowd_mask,unannotated_mask,overlap_mask))
    losses.append(segmentation_loss(seg_true,seg_mask,crowd_mask))
    return losses

with tf.device('/device:GPU:1'):

    train_dataset = DataGenerator(type='TRAIN')
    valid_dataset = DataGenerator(type='VALID')
    
    net = Net()
    
    # variables_name = set()
    # for v in net.variables:
    #     variables_name.add(v.name)
        
    # tvariables_name = set()
    # for v in net.variables:
    #     tvariables_name.add(v.name)
    
    # variables_name.difference(tvariables_name)
    
    optimizer = tf.keras.optimizers.SGD(learning_rate=config.LEARNING_RATE)
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=net)
    manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=10)
    
    epochs = config.NUM_EPOCHS
    batch_size = config.BATCH_SIZE
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,)) 
    
        # train
        for batch_idx in range(train_dataset.datasetlen):
            step = batch_idx
            ckpt.step.assign_add(1)
    
            batch, _ = next(train_dataset.gen_batch(batch_size=batch_size))
            with tf.GradientTape() as tape:
                outputs = net(batch[0])
                losses = get_losses(batch[1:], outputs, step)
                loss = tf.reduce_sum(losses)/batch_size
            
            if step % config.LOG_SCALAR_INTERVAL is 0:
                with writer.as_default():
                    tf.summary.scalar("mean_loss", loss, step=step)
                    
            grads = tape.gradient(loss, net.trainable_variables)
            optimizer.apply_gradients(zip(grads, net.trainable_variables))
            
            # [TODO] add result image visualization
            if batch_idx % config.LOG_IMAGE_INTERVAL is 0:
                # Here is the output map for right shoulder
                Rshoulder_map = outputs[config.KP_MAP_ID][config.TARGET_ID_IN_BATCH][:,:,config.KEYPOINTS.index('Rshoulder')]
                kp_map_image = overlay(batch[config.TARGET_ID_IN_BATCH][config.RAW_ID_IN_BATCH], Rshoulder_map, alpha=0.7)
                H = compute_heatmaps(kp_maps=outputs[config.KP_MAP_ID][config.TARGET_ID_IN_BATCH], short_offsets=outputs[config.SHORT_MAP_ID][config.TARGET_ID_IN_BATCH])
                for i in range(config.NUM_KP):
                    H[:,:,i] = gaussian_filter(H[:,:,i], sigma=2)

                visualized_short_offset = visualize_short_offsets(offsets=outputs[config.SHORT_MAP_ID][config.TARGET_ID_IN_BATCH].numpy(), 
                        heatmaps=H, 
                        keypoint_id='Rshoulder', 
                        img=batch[config.TARGET_ID_IN_BATCH][config.RAW_ID_IN_BATCH], 
                every=8)

                heatmap_image = H[:,:,config.KEYPOINTS.index('Rshoulder')].reshape(1,401,401,1)
                print(f'갑자기 전화온 순간부터 {heatmap_image.shape}')
                print(f'갑자기 전화온 순간부터 {visualized_short_offset.shape}')

                with writer.as_default():
                    kp_map_image_result = tf.summary.image(
                        'train_kp_map_rshoulder', [kp_map_image], step=step, max_outputs=3, description='overlayed kp map and image of rshoulder'
                    )
                    heat_map_image_result = tf.summary.image(
                            'train_heat_map_rshoulder', heatmap_image, step=step, max_outputs=3, description='heatmap and image of rshoulder'
                    )
                    visualized_short_offset_result = tf.summary.image(
                            'train_short_offset_rshoulder', [visualized_short_offset], step=step, max_outputs=3, description='short offset and image of rshoulder'
                    )
     
            if step % 100 is 0:
                writer.flush()
            if step % 1000 is 0:
                save_path = manager.save()
                print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
                print("loss {:1.2f}".format(loss.numpy()))
        
        # validation
        # [TODO] OKS metric is required
        val_metrics = {'loss': []}
        val_dts = []
        for batch_idx in range(valid_dataset.datasetlen):
            step = batch_idx
            batch, img_id = next(valid_dataset.gen_batch(batch_size=batch_size))
            outputs = net(batch[0])
            losses = get_losses(batch[1:], outputs, step, mode='VALID')
            loss = tf.reduce_sum(losses)/batch_size
            val_metrics['loss'].append(loss)
            current_dt = {
                    'image_id': img_id,
                    'category_id': config,
                    'keypoints': [],
                    'score': loss
                    }
        
        with writer.as_default():
            tf.summary.scalar("valid_mean_loss", val_metrics['loss']/valid_dataset.datasetlen, step=epoch)
