%Suleyman_Tugrul_Dincer_ 209535
%Computing_Assignment_4
clear all
data = readtable('FlightDelays.xls','basic',true);
data(:,[3,6,7,12,14,15])=[];

t=[111;125;187;192;224;228;232;267;366;369;371;381;428;441;442;446;451;464;474;477;551;566;575;601;634;643;653;663;705;722;766;793;804;808;814;820;825;849;869;873;898;949;953;957;958;962;965;986;995;1017;1027;1036;1037;1041;1050;1060;1061;1063;1074;1095;1098;1100;1114;1115;1117;1126;1136;1170;1172;1173;1202;1221;1227;1230;1233;1245;1247;1253;1260;1271;1281;1326;1409;1431;1457;1494;1496;1501;1507;1511;1541;1543;1544;1585;1605;1607;1632;1639;1651;1659;1666;1680;1692;1719;1729;1744;1745;1780;1791;1796;1807;1810;1815;1816;1822;1823;1840;1850;1858;1873;1878;1880;1886;1889;1907;1909;1910;1914;1915;1916;1941;1944;1958;1959;1963;1972;1982;2000;2003;2018;2028;2063;2083;2122;2137;2143;2144;2146;2150;2151;2165];

[m,n]=size(data);

data.CARRIER=double(categorical(data.CARRIER));
data.DEST=double(categorical(data.DEST));
data.ORIGIN=double(categorical(data.ORIGIN));
data.FlightStatus=categorical(data.FlightStatus); %2 represents ontime, 1 represents delayed
data.FlightStatus(t)='ontime';
data.FlightStatus=double(data.FlightStatus);
data=table2array(data);
for i=1:2201
    data(i,1)=floor(data(i,1)/100)*60+rem(data(i,1),100);
end

y=[data(:,1),data(:,4),dummyvar(data(:,2)),dummyvar(data(:,3)),dummyvar(data(:,5)),data(:,6),dummyvar(data(:,7)),dummyvar(data(:,8)),data(:,9)];

rng(3);y=array2table(y);
shdata = y(randperm(size(data,1)),:);
valid=shdata(1:(round(m*0.2)),:);
train=shdata(round(m*0.2)+1:m,:);

model = fitctree(train,'y56','SplitCriterion','deviance');
x=max(model.PruneList);error=zeros(x+1,2);
view(model,'Mode','Graph');

for i=0:x
    prunedTree = prune(model, 'Level', i);
    cmvalid = confusionmat(valid.y56,predict(prunedTree,valid));
    e=(cmvalid(1,2)+cmvalid(2,1))/size(valid,1);
    error(i+1,1)=x-i;error(i+1,2)=e;
end
plot(error(:,1),error(:,2));
[Y,I]=min(error(:,2));
prunedTree = prune(model, 'Level', I-1);
cmvalid = confusionmat(valid.y56,predict(prunedTree,valid));
e=(cmvalid(1,2)+cmvalid(2,1))/size(valid,1);
view(prunedTree,'Mode','Graph');