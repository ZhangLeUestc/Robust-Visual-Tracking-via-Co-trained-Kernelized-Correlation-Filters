function [r,p1,p2]=choose_response(r1,r2)

p1=(max(r1(:))-mean(r1(:)))/std(r1(:));
p2=(max(r2(:))-mean(r2(:)))/std(r2(:));
if p1>=p2
    r=r1;
else
    r=r2;

end