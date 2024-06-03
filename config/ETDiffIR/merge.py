class Solution:
    def merge(nums1, m: int, nums2, n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        if nums1 == []:
            return nums2
        if nums2 == []:
            return nums1

        i = 0
        nums3 = []
        length1 = m
        length2 = n
        if length1 < length2:
            minlength = length1
        else:
            minlength = length2
        j = 0
        i = 0
        while 1:
            if nums1[i] < nums2[j]:
                nums3.append(nums1[i])
                i +=1
            else:
                nums3.append(nums2[j])
                j +=1
            if j == minlength or i == minlength:
                break
        if j == minlength:
            nums3.extend(nums1[i::])
        elif i == minlength:
            nums3.extend(nums2[j::])
        nums1 = nums3
        print(nums1)


# Solution.merge(nums1=[1,2,3,0,0,0],m=3, nums2=[2,5,6], n=3)
Solution.merge(nums1=[1],m=1, nums2=[], n=0)