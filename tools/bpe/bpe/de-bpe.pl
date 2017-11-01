#!/usr/bin/perl

if(@ARGV > 0){
   print STDERR "usage: $0 < infile > outfile\n";
   exit(1);
}

@lines=<STDIN>;
foreach my $_(@lines){
    chomp();
    if(m/\|\|\|/){
        @strs = split /\s*\|\|\|\s*/;
        $strs[1] =~ s/\@\@ ([a-z\-\d]+)/$1/gi;
        $strs[1] =~ s/\@\@$/$1/g;
        $strs[1] =~ s/\@\@//g;
        print join(" ||| ", @strs)."\n";
    }else{
        s/\@\@ ([a-z\-\d]+)/$1/g;
        s/\@\@$/$1/g;
        s/\@\@//g;
        print "$_\n";
    }
}


