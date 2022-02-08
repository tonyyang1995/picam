def print_loss(loss_stat, cur_epoch, total_epoch, cur_iter, total_iter):
	message = '\n--------------------[Epoch %d/%d, Batch %d/%d]--------------------\n' % (cur_epoch, total_epoch, cur_iter, total_iter)
	for k, v in loss_stat.items():
		message += '{:>10}\t{:>10.4f}\n'.format(k, v)
	message += '--------------------------------------------------------------------\n'
	return message

def tensorboard_visual(loss_stat, cur_epoch, total_epoch, cur_iter, total_iter, writer, mode='train'):
	counter = total_iter * (cur_epoch) + cur_iter
	for k in loss_stat:
		writer.add_scalar(mode+'/{}'.format(k), loss_stat[k],counter)
		