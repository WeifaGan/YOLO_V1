
import xml.etree.ElementTree as ET
import os

VOC_CLASSES = (   
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')


def parse_xml(filename): 
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        difficult = int(obj.find('difficult').text)
        if difficult == 1:
            # print(filename)
            continue
        obj_struct['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(float(bbox.find('xmin').text)),
                              int(float(bbox.find('ymin').text)),
                              int(float(bbox.find('xmax').text)),
                              int(float(bbox.find('ymax').text))]
        objects.append(obj_struct)

    return objects


def xml_to_txt(xml_dir,train_txt_path):
    """make xml file to txt"""
    with open(train_txt_path,'r',encoding='utf-8') as t:
        for line in t.readlines():
            fixed = line.replace('\n','')
            img_name = fixed+'.jpg'
            xml_name = fixed+'.xml'
            with open('./get_data/voc2012_trainval.txt','a',encoding='utf-8') as f:
                file_path = os.path.join(xml_dir,xml_name)
                objs = parse_xml(file_path)
                f.write(img_name)
                for j in objs:
                    
                    clas = VOC_CLASSES.index(j['name'])
                    bbx = j['bbox']
                    f.write(' '+str(bbx[0])+' '+str(bbx[1])+' '+str(bbx[2])+' '+str(bbx[3])+' '+str(clas)) 
                f.write('\r\n')




if __name__ == "__main__":
    xml_path = '/media/gwf/D1/Dataset/VOC2012/Annotations' 
    train_txt_path = '/media/gwf/D1/Dataset/VOC2012/ImageSets/Main/trainval.txt'
    xml_to_txt(xml_path,train_txt_path)